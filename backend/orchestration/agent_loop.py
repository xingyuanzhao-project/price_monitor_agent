"""
Agentic loop driver: orchestrates an agent node's round-trips to completion.

What it does:
    Implements the AgentLoop class, which owns the agentic cycle for one
    agent execution: call the LLM (one CoreAgent turn), dispatch any tool
    calls through the execution harness, fold results into the
    conversation and node state, evaluate completion, pace the next
    iteration, and repeat — bounded by the workflow-level
    ``max_iterations``.  The loop is orchestration: the core agent it
    drives knows nothing about iteration, tools, or completion.

    Task completion is decided by three layers, evaluated each iteration:
        1. Mechanical (NodeState.check_completion) — did the tool layer
           return non-empty, non-error data?  Advisory: it informs the
           agent via injected status but does not by itself stop the
           loop, so multi-step tool workflows are not cut short.
        2. Agent judgement — the LLM produces a final text response with
           no tool calls.  The model asserts the task is done.
        3. Declarative (NodeState.evaluate_termination_conditions) — the
           node's user-authored termination_conditions are matched
           against the accumulated output; when all are met the loop
           terminates even if the agent would otherwise continue.
    The iteration budget is the hard ceiling: exhausting it without a
    completion verdict raises AgentExecutionError.

Entities in it:
    - AgentExecutionError: Raised when an agent execution fails irrecoverably.
    - AgentLoop: Drives one agent's call → result → answer round-trips.

How used by other modules:
    - backend.orchestration.executor builds an AgentLoop per AGENT node,
      wiring in the CoreAgent, the execution harness's process_response
      as tool_call_handler, the registered NodeState, and the workflow
      config's max_iterations / iteration_sleep.
    - backend.orchestration.group builds an AgentLoop per sub-agent.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from backend.agent.core import CoreAgent
from backend.agent.llm_provider import LLMProviderError
from backend.state import NodeState

_LOGGER = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """
    Raised when an agent fails to complete its execution.

    Description:
        Indicates that the agentic loop could not produce a final result:
        the LLM retries were exhausted, the iteration budget was spent
        without a completion verdict, or a planning/configuration step
        failed.  Carries context about the node and iteration state.

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


class AgentLoop:
    """
    Drives one agent execution's round-trips: call → result → answer.

    Description:
        Owns the agentic cycle around a CoreAgent.  Each iteration runs
        one LLM turn, dispatches tool calls via the tool_call_handler,
        writes conversation and tool records into the NodeState, runs the
        three completion layers, injects a status message for the next
        turn, and sleeps ``iteration_sleep`` seconds before continuing.
        The bound (``max_iterations``) and pacing come from the workflow
        config through the orchestration layer, so the loop is explicit
        at the graph → schema → orchestration level.

    Attributes:
        agent: The CoreAgent performing one LLM turn per iteration.
        tool_call_handler: Async callable dispatching an assistant
            message's tool calls and returning tool result messages.
        node_state: NodeState receiving conversation history, tool call
            records, iteration progress, and the final status.
        termination_conditions: Declarative completion criteria from the
            node config (layer 3).
        max_iterations: Iteration ceiling, from the workflow config.
        iteration_sleep: Seconds to sleep between iterations, from the
            workflow config.

    Methods:
        execute: Run the loop to completion and return the final result.
    """

    def __init__(
        self,
        agent: CoreAgent,
        tool_call_handler: Callable,
        node_state: NodeState,
        termination_conditions: list[str],
        max_iterations: int,
        iteration_sleep: float,
        emit_event: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the AgentLoop with its collaborators and control values.

        Description:
            Stores the agent, tool dispatch callback, state object, and
            the loop control values threaded from the workflow config.

        Params:
            agent (CoreAgent): The single-turn agent to drive.
            tool_call_handler (Callable): Async function to execute tool calls.
                Signature: async (assistant_message: dict) -> list[dict]
            node_state (NodeState): Mutable state for this node.  The loop
                writes conversation history, tool call records, iteration
                progress, and the terminal status to it.
            termination_conditions (list[str]): Declarative completion
                criteria from the node config (layer 3).
            max_iterations (int): Iteration ceiling, from the workflow config.
            iteration_sleep (float): Seconds to sleep between iterations,
                from the workflow config.
            emit_event (Optional[Callable]): Optional async-compatible
                callback receiving event dicts for tracing.
                Signature: async (event: dict) -> None

        Returns:
            None
        """
        self.agent = agent
        self.tool_call_handler = tool_call_handler
        self.node_state = node_state
        self.termination_conditions = termination_conditions
        self.max_iterations = max_iterations
        self.iteration_sleep = iteration_sleep
        self._emit_event = emit_event

    async def execute(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Run the agentic loop to completion.

        Description:
            Iterates up to ``max_iterations`` times.  Each iteration: one
            CoreAgent turn, dispatch of the turn's structured tool_calls
            through the harness, completion evaluation by the three layers
            documented on the module, and status injection for the next
            turn.  Every significant transition is traced via the
            emit_event callback.

        Params:
            messages (list[dict]): Initial conversation messages.
            tools (Optional[list[dict]]): Tool definitions available to
                the LLM, or None.

        Returns:
            dict: Final result with keys:
                - content (str): Final agent output text.
                - iterations (int): Iterations run.
                - tool_calls_made (list[dict]): Accumulated tool-call log.
                - finish_reason (str): LLM finish reason of the final turn.

        Raises:
            AgentExecutionError: If the LLM retries are exhausted or the
                iteration budget is spent without a completion verdict.
        """
        node_id = self.agent.node_id
        self.node_state.conversation_history = list(messages)
        conversation_messages = self.node_state.conversation_history
        tool_calls_record = self.node_state.tool_calls_record

        # Accumulated agent output + tool results, matched by the
        # declarative termination layer.
        completion_text_parts: list[str] = []
        iterations_completed = 0

        for iteration_index in range(self.max_iterations):
            iterations_completed = iteration_index + 1
            self.node_state.iteration = iterations_completed

            _LOGGER.info("[%s] Iteration %d/%d — calling LLM",
                         node_id, iterations_completed, self.max_iterations)

            await self._emit({
                "type": "iteration_started",
                "iteration": iterations_completed,
                "max_iterations": self.max_iterations,
                "message_count": len(conversation_messages),
                "task_status": self.node_state.status.value,
            })

            try:
                turn = await self.agent.execute_turn(conversation_messages, tools)
            except LLMProviderError as provider_error:
                await self._transition_state(
                    "fail", "All LLM call attempts failed (retries exhausted)",
                    reason="llm_retries_exhausted",
                )
                raise AgentExecutionError(
                    f"All LLM call attempts failed (retries exhausted): {provider_error}",
                    node_id=node_id,
                    iterations_completed=iterations_completed,
                ) from provider_error

            assistant_message = turn["assistant_message"]
            finish_reason = turn["finish_reason"]

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

            # --- Tool call path ---
            result_messages = await self._dispatch_tool_calls(
                assistant_message, has_native_tool_calls,
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
            for result_message in result_messages:
                if result_message.get("content"):
                    completion_text_parts.append(str(result_message["content"]))
            accumulated_output = "\n".join(completion_text_parts)

            # --- Layer 1: mechanical verdict (advisory) ---
            verdict = self.node_state.check_completion(result_messages)
            # --- Layer 3: declarative termination conditions ---
            termination = self.node_state.evaluate_termination_conditions(
                self.termination_conditions, accumulated_output,
            )
            await self._emit({
                "type": "completion_check",
                "iteration": iterations_completed,
                "verdict": verdict,
                "termination": termination,
            })
            self._inject_status_context(
                conversation_messages, verdict, termination, iterations_completed,
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

            if self.iteration_sleep > 0:
                await asyncio.sleep(self.iteration_sleep)

        await self._transition_state(
            "fail", f"Maximum iterations ({self.max_iterations}) reached",
            reason="max_iterations_reached",
        )
        raise AgentExecutionError(
            f"Maximum iterations ({self.max_iterations}) reached without termination",
            node_id=node_id,
            iterations_completed=iterations_completed,
        )

    # -- iteration internals --------------------------------------------------

    async def _dispatch_tool_calls(
        self,
        assistant_message: dict,
        has_native_tool_calls: bool,
        conversation_messages: list[dict],
        tool_calls_record: list[dict],
        iteration: int,
    ) -> Optional[list[dict]]:
        """Dispatch the assistant message's tool calls and append results.

        Description:
            The structured ``tool_calls`` block is the single authoritative
            signal that the agent requested tools this turn.  When present,
            the full assistant message is handed to the tool_call_handler
            (the execution harness's process_response) for dispatch; result
            messages are appended to the conversation and recorded in the
            node state's tool call log.  A message without ``tool_calls``
            is a final answer — never re-interpreted as a tool request.

        Params:
            assistant_message (dict): The LLM response message.
            has_native_tool_calls (bool): Whether the message contains tool_calls.
            conversation_messages (list[dict]): The live conversation list (mutated).
            tool_calls_record (list[dict]): Accumulated tool call log (mutated).
            iteration (int): Current iteration number for tracing.

        Returns:
            Optional[list[dict]]: Tool result messages, or None if no tool
                calls occurred this turn.
        """
        if not has_native_tool_calls:
            return None

        result_messages = await self.tool_call_handler(assistant_message)

        conversation_messages.extend(result_messages)

        await self._emit({
            "type": "tool_results_appended",
            "iteration": iteration,
            "result_count": len(result_messages),
            "results_preview": [
                {
                    "role": result_message.get("role"),
                    "tool_call_id": result_message.get("tool_call_id", ""),
                    "content_length": len(result_message.get("content", "")),
                    "content_preview": result_message.get("content", "")[:500],
                }
                for result_message in result_messages
            ],
        })

        for result_message in result_messages:
            tool_calls_record.append({
                "result": result_message.get("content", ""),
            })

        return result_messages

    def _inject_status_context(
        self,
        conversation_messages: list[dict],
        verdict: dict[str, Any],
        termination: dict[str, Any],
        iteration: int,
    ) -> None:
        """Inject task status, mechanical verdict, and explicit done-criteria.

        Description:
            Appends a system-role message so the LLM (layer 2) can see
            where the task stands, what the mechanical check found, and
            which declared termination conditions remain unmet — and
            decide whether to continue or produce a final answer.

        Params:
            conversation_messages (list[dict]): The live conversation list (mutated).
            verdict (dict[str, Any]): The mechanical completion verdict (layer 1).
            termination (dict[str, Any]): The declarative termination verdict (layer 3).
            iteration (int): Current iteration number.

        Returns:
            None
        """
        status_text = (
            f"[Task status — iteration {iteration}/{self.max_iterations}]\n"
            f"Task: {self.agent.label}\n"
            f"Completion check: successful_results={verdict['successful_count']}, "
            f"failed={verdict['failed_count']}, "
            f"total_tool_calls={verdict['total_tool_calls']}\n"
        )
        if termination["active"]:
            if termination["unmet"]:
                status_text += (
                    "Declared completion criteria not yet detected in the "
                    f"output: {termination['unmet']}.\n"
                )
            else:
                status_text += "All declared completion criteria are met.\n"
        status_text += f"Status: {self.node_state.status.value}\n"

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

    # -- completion and tracing ------------------------------------------------

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

        Description:
            Single return point for both completion layers (agent
            judgement and declarative termination), so the completion
            side effects are not duplicated.

        Params:
            content (str): Final agent output text.
            iterations_completed (int): Iterations run.
            finish_reason (str): LLM finish reason of the final turn.
            tool_calls_record (list[dict]): Accumulated tool-call log.
            has_tool_calls (bool): Whether the final iteration made tool calls.
            reason (str): Optional state-change reason for the trace.

        Returns:
            dict: The loop's final result (see ``execute``).
        """
        await self._transition_state("complete", content[:500], reason=reason)
        _LOGGER.info("[%s] Agent finished — reason=%s, iterations=%d, content_length=%d",
                     self.agent.node_id, reason or finish_reason,
                     iterations_completed, len(content))
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

    async def _transition_state(
        self, action: str, summary: str, *, reason: str = "",
    ) -> None:
        """Transition node state and emit a traced state_change event.

        Description:
            Reads the current status from the node state before mutating,
            so the trace shows the actual old value rather than a
            hardwired assumption.

        Params:
            action (str): ``"complete"`` or ``"fail"``.
            summary (str): Text passed to ``state.complete()`` or ``state.fail()``.
            reason (str): Optional reason string for the trace event.

        Returns:
            None
        """
        old_status = self.node_state.status.value
        if action == "complete":
            self.node_state.complete(summary=summary)
        else:
            self.node_state.fail(error=summary)
        event: dict[str, Any] = {
            "type": "state_change",
            "field": "status",
            "old_value": old_status,
            "new_value": self.node_state.status.value,
        }
        if reason:
            event["reason"] = reason
        await self._emit(event)

    async def _emit(self, event: dict[str, Any]) -> None:
        """Emit a trace event through the injected callback.

        Description:
            Forwards *event* to the emit_event callback when one was
            provided, awaiting it if it is a coroutine.  ``node_id`` is
            injected automatically.

        Params:
            event (dict[str, Any]): Event payload to emit.

        Returns:
            None
        """
        if self._emit_event is None:
            return
        event["node_id"] = self.agent.node_id
        result = self._emit_event(event)
        if asyncio.iscoroutine(result):
            await result


def _synthesize_content(result_messages: list[dict]) -> str:
    """Build a final content string from tool results when the LLM
    produced no text alongside its tool calls.

    Description:
        Only includes messages with substantive content (length > 2).
        By the time results reach the loop, they have already survived
        the execution harness without exception, so all are valid results.

    Params:
        result_messages (list[dict]): Tool result messages from the final
            iteration.

    Returns:
        str: Concatenated tool result content.
    """
    parts = []
    for result_message in result_messages:
        content = result_message.get("content", "")
        if len(content) > 2:
            parts.append(content)
    return "\n".join(parts)
