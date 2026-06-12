"""
Core agent implementation with agentic loop and retry logic.

What it does:
    Implements the CoreAgent class which runs an agentic loop: calling the LLM,
    processing tool calls via a handler callback, looping until termination
    conditions are met or max iterations reached, with exponential backoff
    retries on failures.

Entities in it:
    - AgentExecutionError: Exception raised when agent execution fails irrecoverably.
    - CoreAgent: The main agent class implementing the agentic execution loop.

How used by other modules:
    - backend.agent.group creates CoreAgent instances for each sub-agent in a group.
    - The orchestration engine creates CoreAgent for standalone AGENT nodes.
    - Tool calls during the loop are dispatched via the tool_call_handler callback,
      which typically resolves tools from backend.tools.registry.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.schema.models import NodeConfig
from backend.state import NodeState

_LOGGER = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """
    Raised when an agent fails to complete its execution.

    Description:
        Indicates that the agentic loop could not produce a final result
        after exhausting all retries and fallback options. Carries context
        about the node, iteration state, and underlying cause.

    Attributes:
        message: Human-readable description of the failure.
        node_id: Identifier of the agent node that failed.
        iterations_completed: Number of loop iterations completed before failure.
    """

    def __init__(self, message: str, node_id: str, iterations_completed: int) -> None:
        """
        Initialize with failure context.

        Description:
            Stores the node identifier and iteration count for diagnostics.

        Params:
            message (str): Description of the execution failure.
            node_id (str): The agent node's identifier.
            iterations_completed (int): How many iterations ran before failure.

        Returns:
            None
        """
        self.message = message
        self.node_id = node_id
        self.iterations_completed = iterations_completed
        super().__init__(f"Agent '{node_id}' failed after {iterations_completed} iterations: {message}")


class CoreAgent:
    """
    Implements the agentic execution loop for a single agent node.

    Description:
        Repeatedly calls the LLM, processes any tool_calls by dispatching
        them to the tool_call_handler, appends results to the conversation,
        and loops until completion is decided or the iteration budget is
        spent. The loop bound (``max_iterations``) and inter-iteration
        sleep are not owned by the agent — they are supplied by the
        orchestration layer from the workflow config, so the loop is
        explicit at the graph→schema→orchestration level rather than
        hardwired per node. Includes retry with exponential backoff.

    Task completion is decided by three layers, evaluated each iteration:
        1. Mechanical (NodeState.check_completion) — did the tool layer
           return non-empty, non-error data? Advisory: it informs the
           agent (via injected status) but does not by itself stop the
           loop, so multi-step tool workflows are not cut short.
        2. Agent judgement — the LLM produces a final text response with
           no tool calls. The model asserts the task is done.
        3. Declarative (NodeState.evaluate_termination_conditions) — the
           node's user-authored termination_conditions are matched against
           the accumulated output; when all are met the loop terminates
           even if the agent would otherwise continue.
    The iteration budget is the hard ceiling: exhausting it without a
    completion verdict raises AgentExecutionError.

    Attributes:
        node_id: Unique identifier for this agent instance.
        label: Human-readable label for this agent.
        config: NodeConfig with model, retry, and termination settings.
        llm_provider: LLMProvider instance for API calls.
        tool_call_handler: Async callable that executes a tool call and returns result.
        emit_event: Optional async-compatible callback receiving event dicts.
            Used for ALL trace events from the agent layer.
        state: Optional NodeState for centralized state tracking.

    Methods:
        execute: Run the agentic loop and return the final result.
    """

    def __init__(
        self,
        node_id: str,
        label: str,
        config: NodeConfig,
        llm_provider: LLMProvider,
        tool_call_handler: Callable,
        emit_event: Optional[Callable] = None,
        node_state: Optional[NodeState] = None,
    ) -> None:
        """
        Initialize the CoreAgent with its configuration and dependencies.

        Description:
            Stores all parameters needed for the agentic loop execution.

        Params:
            node_id (str): Unique agent node identifier.
            label (str): Human-readable agent label.
            config (NodeConfig): Agent configuration from the workflow schema.
            llm_provider (LLMProvider): Provider for LLM API calls.
            tool_call_handler (Callable): Async function to execute tool calls.
                Signature: async (assistant_message: dict) -> list[dict]
            emit_event (Optional[Callable]): Optional async-compatible callback
                receiving event dicts for tracing.
                Signature: async (event: dict) -> None
            node_state (Optional[NodeState]): Mutable state object for this
                node.  If provided, the agent writes conversation history,
                tool call records, and iteration progress to it.

        Returns:
            None
        """
        self.node_id = node_id
        self.label = label
        self.config = config
        self.llm_provider = llm_provider
        self.tool_call_handler = tool_call_handler
        self._emit_event = emit_event
        self.state = node_state

    async def _emit(self, event: dict[str, Any]) -> None:
        """Emit a trace event through the injected callback.

        All agent-layer events flow through this single method.

        Args:
            event: Event dict. ``node_id`` is injected automatically.
        """
        if self._emit_event is None:
            return
        event["node_id"] = self.node_id
        result = self._emit_event(event)
        if asyncio.iscoroutine(result):
            await result

    async def _transition_state(
        self, action: str, summary: str, *, reason: str = "",
    ) -> None:
        """Transition node state and emit a traced state_change event.

        Reads the current status from ``self.state`` before mutating, so
        the trace shows the actual old value rather than a hardwired
        assumption.

        Args:
            action: ``"complete"`` or ``"fail"``.
            summary: Text passed to ``state.complete()`` or ``state.fail()``.
            reason: Optional reason string for the trace event.
        """
        if self.state is None:
            return
        old_status = self.state.status.value
        if action == "complete":
            self.state.complete(summary=summary)
        else:
            self.state.fail(error=summary)
        event: dict[str, Any] = {
            "type": "state_change",
            "field": "status",
            "old_value": old_status,
            "new_value": self.state.status.value,
        }
        if reason:
            event["reason"] = reason
        await self._emit(event)

    async def execute(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        *,
        max_iterations: int,
        iteration_sleep: float,
    ) -> dict:
        """Run the agentic execution loop.

        Calls the LLM iteratively. Hands the full assistant_message to the
        execution harness (via tool_call_handler) which produces the ToolRequest
        structure and dispatches to tools. The harness handles both native
        tool_calls and non-native text requests.

        The loop is bounded and paced by orchestration, not by the agent:

        Args:
            messages: Initial conversation messages.
            tools: Tool definitions available to the LLM, or None.
            max_iterations: Iteration ceiling, from the workflow config.
            iteration_sleep: Seconds to sleep between iterations, from the
                workflow config.

        Completion follows the three layers documented on the class.
        Traces every significant transition via ``_emit``.
        """
        if self.state is not None:
            self.state.conversation_history = list(messages)
            conversation_messages = self.state.conversation_history
            tool_calls_record = self.state.tool_calls_record
        else:
            conversation_messages = list(messages)
            tool_calls_record = []

        # Accumulated agent output + tool results, matched by the
        # declarative termination layer.
        completion_text_parts: list[str] = []
        iterations_completed = 0

        for iteration_index in range(max_iterations):
            iterations_completed = iteration_index + 1
            if self.state is not None:
                self.state.iteration = iterations_completed

            _LOGGER.info("[%s] Iteration %d/%d — calling LLM",
                         self.node_id, iterations_completed, max_iterations)

            await self._emit({
                "type": "iteration_started",
                "iteration": iterations_completed,
                "max_iterations": max_iterations,
                "message_count": len(conversation_messages),
                "task_status": self.state.status.value if self.state else "unknown",
            })

            llm_response = await self._call_llm_with_retries(
                self.llm_provider, conversation_messages, tools
            )

            if llm_response is None:
                _LOGGER.error("[%s] All LLM retries exhausted", self.node_id)
                await self._transition_state(
                    "fail", "All LLM call attempts failed (retries exhausted)",
                    reason="llm_retries_exhausted",
                )
                raise AgentExecutionError(
                    "All LLM call attempts failed (retries exhausted)",
                    node_id=self.node_id,
                    iterations_completed=iterations_completed,
                )

            choice = llm_response["choices"][0]
            assistant_message = choice["message"]
            finish_reason = choice.get("finish_reason", "stop")

            conversation_messages.append(assistant_message)
            if assistant_message.get("content"):
                completion_text_parts.append(assistant_message["content"])

            has_native_tool_calls = bool(assistant_message.get("tool_calls"))

            await self._emit({
                "type": "llm_response",
                "iteration": iterations_completed,
                "finish_reason": finish_reason,
                "has_tool_calls": has_native_tool_calls,
                "content_length": len(assistant_message.get("content") or ""),
            })

            # --- Tool call path (native or non-native) ---
            result_messages = await self._process_tool_calls(
                assistant_message, has_native_tool_calls, tools,
                conversation_messages, tool_calls_record, iterations_completed,
            )

            # --- Layer 2: agent judgement — no tool calls means done ---
            if result_messages is None:
                content = assistant_message.get("content", "")
                return await self._finish(
                    content, iterations_completed, finish_reason,
                    tool_calls_record, has_tool_calls=False,
                )

            # Tools ran this iteration. Fold their results into the text the
            # declarative layer matches against.
            for msg in result_messages:
                if msg.get("content"):
                    completion_text_parts.append(str(msg["content"]))
            accumulated_output = "\n".join(completion_text_parts)

            # --- Layer 1: mechanical verdict (advisory) ---
            verdict = self.state.check_completion(result_messages)
            # --- Layer 3: declarative termination conditions ---
            termination = self.state.evaluate_termination_conditions(
                self.config.termination_conditions, accumulated_output,
            )
            await self._emit({
                "type": "completion_check",
                "iteration": iterations_completed,
                "verdict": verdict,
                "termination": termination,
            })
            self._inject_status_context(
                conversation_messages, verdict, termination, iterations_completed,
                max_iterations,
            )

            if termination["active"] and termination["satisfied"]:
                content = assistant_message.get("content") or _synthesize_content(result_messages)
                return await self._finish(
                    content, iterations_completed, finish_reason,
                    tool_calls_record, has_tool_calls=True,
                    reason="termination_conditions_met",
                )

            await self._emit({
                "type": "node_output",
                "chunk": {
                    "iteration": iterations_completed,
                    "content": assistant_message.get("content"),
                    "finish_reason": finish_reason,
                    "has_tool_calls": True,
                },
            })

            if iteration_sleep > 0:
                await asyncio.sleep(iteration_sleep)

        await self._transition_state(
            "fail", f"Maximum iterations ({max_iterations}) reached",
            reason="max_iterations_reached",
        )
        raise AgentExecutionError(
            f"Maximum iterations ({max_iterations}) reached without termination",
            node_id=self.node_id,
            iterations_completed=iterations_completed,
        )

    async def _finish(
        self,
        content: str,
        iterations_completed: int,
        finish_reason: str,
        tool_calls_record: list[dict],
        *,
        has_tool_calls: bool,
        reason: str = "",
    ) -> dict:
        """Complete the node, emit the final node_output, and build the result.

        Single return point for both completion layers (agent judgement
        and declarative termination), so the completion side effects are
        not duplicated.

        Args:
            content: Final agent output text.
            iterations_completed: Iterations run.
            finish_reason: LLM finish reason of the final response.
            tool_calls_record: Accumulated tool-call log.
            has_tool_calls: Whether the final iteration made tool calls.
            reason: Optional state-change reason for the trace.
        """
        await self._transition_state("complete", content[:500], reason=reason)
        _LOGGER.info("[%s] Agent finished — reason=%s, iterations=%d, content_length=%d",
                     self.node_id, reason or finish_reason, iterations_completed, len(content))
        await self._emit({
            "type": "node_output",
            "chunk": {
                "iteration": iterations_completed,
                "content": content,
                "finish_reason": finish_reason,
                "has_tool_calls": has_tool_calls,
            },
        })
        return {
            "content": content,
            "iterations": iterations_completed,
            "tool_calls_made": tool_calls_record,
            "finish_reason": finish_reason,
        }

    async def _call_llm_with_retries(
        self,
        provider: LLMProvider,
        messages: list[dict],
        tools: Optional[list[dict]],
    ) -> Optional[dict]:
        """Call the LLM with exponential backoff retries.

        Attempts the LLM call up to ``config.retries + 1`` times, with
        exponential backoff between attempts.  Each retry is traced so the
        event log shows every attempt.

        Args:
            provider: The LLM provider to call.
            messages: Conversation messages.
            tools: Tool definitions.

        Returns:
            The LLM response dict, or None if all attempts failed.
        """
        last_error: Optional[LLMProviderError] = None

        tc = self.config.tool_choice if tools else None
        ptc = self.config.parallel_tool_calls if tools else None
        max_attempts = self.config.retries + 1

        for attempt_index in range(max_attempts):
            try:
                response = await provider.complete(
                    messages=messages,
                    tools=tools,
                    response_format=self.config.response_format,
                    tool_choice=tc,
                    parallel_tool_calls=ptc,
                )
                return response
            except LLMProviderError as provider_error:
                last_error = provider_error
                if attempt_index < self.config.retries:
                    backoff_seconds = self.config.retry_waiting_time ** attempt_index
                    _LOGGER.warning("[%s] LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                                    self.node_id, attempt_index + 1,
                                    max_attempts, provider_error,
                                    backoff_seconds)
                    await self._emit({
                        "type": "llm_retry",
                        "attempt": attempt_index + 1,
                        "max_attempts": max_attempts,
                        "error": str(provider_error),
                        "backoff_seconds": backoff_seconds,
                    })
                    await asyncio.sleep(backoff_seconds)

        _ = last_error
        return None

    async def _process_tool_calls(
        self,
        assistant_message: dict,
        has_native_tool_calls: bool,
        tools: Optional[list[dict]],
        conversation_messages: list[dict],
        tool_calls_record: list[dict],
        iteration: int,
    ) -> Optional[list[dict]]:
        """Dispatch tool calls (native or non-native) and append results.

        Returns the result messages if tool calls were processed, or None
        if no tool calls were made.

        Args:
            assistant_message: The LLM response message.
            has_native_tool_calls: Whether the message contains tool_calls.
            tools: Tool definitions (None if no tools available).
            conversation_messages: The live conversation list (mutated).
            tool_calls_record: Accumulated tool call log (mutated).
            iteration: Current iteration number for tracing.

        Returns:
            List of tool result messages, or None if no tool calls occurred.
        """
        if has_native_tool_calls:
            call_type = "native"
        elif tools and assistant_message.get("content"):
            call_type = "non_native"
        else:
            return None

        result_messages = await self.tool_call_handler(assistant_message)

        if call_type == "non_native" and not result_messages:
            return None

        conversation_messages.extend(result_messages)

        await self._emit({
            "type": "tool_results_appended",
            "iteration": iteration,
            "call_type": call_type,
            "result_count": len(result_messages),
            "results_preview": [
                {
                    "role": msg.get("role"),
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content_length": len(msg.get("content", "")),
                    "content_preview": msg.get("content", "")[:500],
                }
                for msg in result_messages
            ],
        })

        for msg in result_messages:
            tool_calls_record.append({
                "type": call_type,
                "result": msg.get("content", ""),
            })

        return result_messages

    def _inject_status_context(
        self,
        conversation_messages: list[dict],
        verdict: dict[str, Any],
        termination: dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> None:
        """Inject task status, mechanical verdict, and explicit done-criteria.

        Appended as a system-role message so the LLM (layer 2) can see
        where the task stands, what the mechanical check found, and which
        declared termination conditions remain unmet — and decide whether
        to continue or produce a final answer.

        Args:
            conversation_messages: The live conversation list (mutated).
            verdict: The mechanical completion verdict (layer 1).
            termination: The declarative termination verdict (layer 3).
            iteration: Current iteration number.
            max_iterations: Iteration ceiling from the workflow config.
        """
        status_text = (
            f"[Task status — iteration {iteration}/{max_iterations}]\n"
            f"Task: {self.label}\n"
            f"Completion check: successful_results={verdict['successful_count']}, "
            f"failed={verdict['failed_count']}, "
            f"total_tool_calls={verdict['total_tool_calls']}\n"
        )
        if termination["active"]:
            if termination["unmet"]:
                status_text += (
                    "Termination conditions not yet met: "
                    f"{termination['unmet']}. Continue until they are satisfied.\n"
                )
            else:
                status_text += "All termination conditions met.\n"
        if self.state is not None:
            status_text += f"Status: {self.state.status.value}\n"

        if verdict["has_successful_result"]:
            status_text += (
                "Tool calls returned data successfully. "
                "If the task objective is met, produce a final text response. "
                "If more data is needed, continue with additional tool calls."
            )
        else:
            status_text += (
                "No successful tool results yet. "
                "Review the errors and adjust your approach."
            )

        conversation_messages.append({
            "role": "system",
            "content": status_text,
        })


def _synthesize_content(result_messages: list[dict]) -> str:
    """Build a final content string from tool results when the LLM
    produced no text alongside its tool calls.

    Only includes messages with substantive content (length > 2).
    By the time results reach the agent, they have already survived
    the execution harness without exception, so all are valid results.

    Args:
        result_messages: Tool result messages from the final iteration.

    Returns:
        Concatenated tool result content.
    """
    parts = []
    for msg in result_messages:
        content = msg.get("content", "")
        if len(content) > 2:
            parts.append(content)
    return "\n".join(parts)

