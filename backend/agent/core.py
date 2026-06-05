"""
Core agent implementation with agentic loop, retry logic, and fallback.

What it does:
    Implements the CoreAgent class which runs an agentic loop: calling the LLM,
    processing tool calls via a handler callback, looping until termination
    conditions are met or max iterations reached, with exponential backoff
    retries and optional fallback model switching on persistent failures.

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
import json
from typing import Any, Callable, Optional

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.schema.models import NodeConfig


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
        """
        Run the agentic execution loop.

        Description:
            Iteratively calls the LLM, processes tool_calls, and accumulates
            conversation history until termination. Implements retry with
            exponential backoff on LLM errors and switches to fallback model
            if retries are exhausted.

        Params:
            messages (list[dict]): Initial conversation messages.
            tools (Optional[list[dict]]): Tool definitions for the LLM.

        Returns:
            dict: Result dictionary with keys:
                - content (str): Final text response from the agent.
                - iterations (int): Number of loop iterations completed.
                - tool_calls_made (list[dict]): Record of all tool calls made.
                - finish_reason (str): Why the loop terminated.

        Raises:
            AgentExecutionError: If all retries and fallback are exhausted.
        """
        conversation_messages = list(messages)
        tool_calls_record: list[dict] = []
        iterations_completed = 0
        current_provider = self.llm_provider
        using_fallback = False

        for iteration_index in range(self.config.max_iterations):
            iterations_completed = iteration_index + 1

            llm_response = await self._call_llm_with_retries(
                current_provider, conversation_messages, tools
            )

            if llm_response is None and not using_fallback and self.config.fallback_model_id:
                current_provider = self._create_fallback_provider()
                using_fallback = True
                llm_response = await self._call_llm_with_retries(
                    current_provider, conversation_messages, tools
                )

            if llm_response is None:
                raise AgentExecutionError(
                    "All LLM call attempts failed (retries and fallback exhausted)",
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

            if assistant_message.get("tool_calls"):
                tool_results = await self._process_tool_calls(
                    assistant_message["tool_calls"], tool_calls_record
                )
                conversation_messages.extend(tool_results)

                if self._check_termination_conditions(assistant_message, tool_calls_record):
                    content = assistant_message.get("content", "")
                    return {
                        "content": content,
                        "iterations": iterations_completed,
                        "tool_calls_made": tool_calls_record,
                        "finish_reason": "termination_condition",
                    }
                continue

            content = assistant_message.get("content", "")
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
                    backoff_seconds = self.config.backoff_multiplier ** attempt_index
                    await asyncio.sleep(backoff_seconds)

        _ = last_error
        return None

    async def _process_tool_calls(
        self, tool_calls: list[dict], tool_calls_record: list[dict]
    ) -> list[dict]:
        """
        Process tool calls from the LLM response by invoking the handler.

        Description:
            Iterates over tool_calls, invokes each via the tool_call_handler,
            records the call and result, and constructs tool response messages.

        Params:
            tool_calls (list[dict]): Tool call objects from the LLM response.
            tool_calls_record (list[dict]): Accumulator for all tool calls made.

        Returns:
            list[dict]: Tool response messages to append to conversation.

        Raises:
            AgentExecutionError: If a tool call handler raises an exception.
        """
        tool_response_messages: list[dict] = []

        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_data = tool_call["function"]
            tool_name = function_data["name"]
            tool_arguments_str = function_data.get("arguments", "{}")

            try:
                tool_arguments = json.loads(tool_arguments_str)
            except json.JSONDecodeError as parse_error:
                raise AgentExecutionError(
                    f"Failed to parse tool call arguments for '{tool_name}': {parse_error}. "
                    f"Raw arguments: {tool_arguments_str}",
                    node_id=self.node_id,
                    iterations_completed=0,
                ) from parse_error

            tool_result = await self.tool_call_handler(tool_name, tool_arguments)

            tool_calls_record.append({
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "result": tool_result,
            })

            result_content = json.dumps(tool_result, default=str) if not isinstance(tool_result, str) else tool_result

            tool_response_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            })

        return tool_response_messages

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

    def _create_fallback_provider(self) -> LLMProvider:
        """
        Create a new LLMProvider instance using the fallback model.

        Description:
            Constructs a provider with the same API key and settings but
            using the fallback_model_id from config.

        Params:
            None

        Returns:
            LLMProvider: A new provider instance configured with the fallback model.
        """
        return LLMProvider(
            api_key=self.llm_provider.api_key,
            model_id=self.config.fallback_model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            base_url=self.llm_provider.base_url,
        )
