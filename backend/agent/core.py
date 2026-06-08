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
        and loops until a termination condition is met, max_iterations is
        reached, or the LLM produces a final text response without tool_calls.
        Includes retry with exponential backoff and optional fallback model.

    Attributes:
        node_id: Unique identifier for this agent instance.
        label: Human-readable label for this agent.
        config: NodeConfig with model, retry, and termination settings.
        llm_provider: LLMProvider instance for API calls.
        tool_call_handler: Async callable that executes a tool call and returns result.
        stream_callback: Optional async callable receiving streaming chunks.

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
        stream_callback: Optional[Callable] = None,
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
                Signature: async (tool_name: str, tool_args: dict) -> Any
            stream_callback (Optional[Callable]): Optional async callback for streaming.
                Signature: async (chunk: dict) -> None

        Returns:
            None
        """
        self.node_id = node_id
        self.label = label
        self.config = config
        self.llm_provider = llm_provider
        self.tool_call_handler = tool_call_handler
        self.stream_callback = stream_callback

    async def execute(self, messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
        """Run the agentic execution loop.

        Calls the LLM iteratively. Hands the full assistant_message to the
        execution harness (via tool_call_handler) which produces the ToolRequest
        structure and dispatches to tools. The harness handles both native
        tool_calls and non-native text requests.
        """
        conversation_messages = list(messages)
        tool_calls_record: list[dict] = []
        iterations_completed = 0

        for iteration_index in range(self.config.max_iterations):
            iterations_completed = iteration_index + 1
            _LOGGER.info("[%s] Iteration %d/%d — calling LLM",
                         self.node_id, iterations_completed, self.config.max_iterations)

            llm_response = await self._call_llm_with_retries(
                self.llm_provider, conversation_messages, tools
            )

            if llm_response is None:
                _LOGGER.error("[%s] All LLM retries exhausted", self.node_id)
                raise AgentExecutionError(
                    "All LLM call attempts failed (retries exhausted)",
                    node_id=self.node_id,
                    iterations_completed=iterations_completed,
                )

            choice = llm_response["choices"][0]
            assistant_message = choice["message"]
            finish_reason = choice.get("finish_reason", "stop")

            conversation_messages.append(assistant_message)

            if self.stream_callback is not None:
                stream_result = self.stream_callback({
                    "node_id": self.node_id,
                    "iteration": iteration_index,
                    "content": assistant_message.get("content", ""),
                    "finish_reason": finish_reason,
                    "has_tool_calls": bool(assistant_message.get("tool_calls")),
                })
                if asyncio.iscoroutine(stream_result):
                    await stream_result

            # --- Hand full assistant_message to execution harness ---
            # The harness produces {native_tool_request, non_native_tool_request}
            # and dispatches to the tool support layer.
            has_native_tool_calls = bool(assistant_message.get("tool_calls"))

            if has_native_tool_calls:
                # Native path: LLM produced structured tool_calls
                _LOGGER.info("[%s] Native tool calls detected", self.node_id)
                result_messages = await self.tool_call_handler(assistant_message)
                conversation_messages.extend(result_messages)

                for msg in result_messages:
                    tool_calls_record.append({"type": "native", "result": msg.get("content", "")})

                if self._check_termination_conditions(assistant_message, tool_calls_record):
                    return {
                        "content": assistant_message.get("content", ""),
                        "iterations": iterations_completed,
                        "tool_calls_made": tool_calls_record,
                        "finish_reason": "termination_condition",
                    }
                if self.config.iteration_sleep > 0:
                    await asyncio.sleep(self.config.iteration_sleep)
                continue

            elif tools and assistant_message.get("content"):
                # Non-native path: LLM produced text, tools are available.
                # Pass to harness — tool layer tries parse_request.
                result_messages = await self.tool_call_handler(assistant_message)
                if result_messages:
                    _LOGGER.info("[%s] Non-native tool match found", self.node_id)
                    conversation_messages.extend(result_messages)

                    for msg in result_messages:
                        tool_calls_record.append({"type": "non_native", "result": msg.get("content", "")})

                    if self.config.iteration_sleep > 0:
                        await asyncio.sleep(self.config.iteration_sleep)
                    continue
                # No tool matched — fall through to treat as final answer

            content = assistant_message.get("content", "")
            _LOGGER.info("[%s] Agent finished — reason=%s, iterations=%d, content_length=%d",
                         self.node_id, finish_reason, iterations_completed, len(content))
            return {
                "content": content,
                "iterations": iterations_completed,
                "tool_calls_made": tool_calls_record,
                "finish_reason": finish_reason,
            }

        raise AgentExecutionError(
            f"Maximum iterations ({self.config.max_iterations}) reached without termination",
            node_id=self.node_id,
            iterations_completed=iterations_completed,
        )

    async def _call_llm_with_retries(
        self,
        provider: LLMProvider,
        messages: list[dict],
        tools: Optional[list[dict]],
    ) -> Optional[dict]:
        """
        Call the LLM with exponential backoff retries.

        Description:
            Attempts the LLM call up to (config.retries + 1) times, waiting
            with exponential backoff between attempts. Returns None if all
            attempts fail.

        Params:
            provider (LLMProvider): The LLM provider to call.
            messages (list[dict]): Conversation messages.
            tools (Optional[list[dict]]): Tool definitions.

        Returns:
            Optional[dict]: The LLM response dict, or None if all attempts failed.
        """
        last_error: Optional[LLMProviderError] = None

        for attempt_index in range(self.config.retries + 1):
            try:
                response = await provider.complete(
                    messages=messages,
                    tools=tools,
                    response_format=self.config.response_format,
                )
                return response
            except LLMProviderError as provider_error:
                last_error = provider_error
                if attempt_index < self.config.retries:
                    backoff_seconds = self.config.retry_waiting_time ** attempt_index
                    _LOGGER.warning("[%s] LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                                    self.node_id, attempt_index + 1,
                                    self.config.retries + 1, provider_error,
                                    backoff_seconds)
                    await asyncio.sleep(backoff_seconds)

        _ = last_error
        return None

    def _check_termination_conditions(
        self, assistant_message: dict, tool_calls_record: list[dict]
    ) -> bool:
        """
        Check if any configured termination conditions are met.

        Description:
            Evaluates the termination_conditions from config against the
            current state of the conversation and tool call history.

        Params:
            assistant_message (dict): The latest assistant message.
            tool_calls_record (list[dict]): All tool calls made so far.

        Returns:
            bool: True if a termination condition is satisfied.
        """
        if not self.config.termination_conditions:
            return False

        content = assistant_message.get("content", "") or ""
        for condition in self.config.termination_conditions:
            if condition in content:
                return True

        return False

