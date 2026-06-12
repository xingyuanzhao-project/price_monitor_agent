"""
Core agent: a single LLM turn with retry logic.

What it does:
    Implements the CoreAgent class, which performs exactly one LLM
    request–response turn: it sends the conversation to the provider with
    the node's generation settings (response_format, tool_choice,
    parallel_tool_calls) and returns the assistant message.  Failed calls
    are retried with exponential backoff; when all attempts are spent the
    last provider error is re-raised so the caller sees the real failure.

    The agentic loop — repeating turns, dispatching tool calls, deciding
    completion, pacing iterations — is NOT here.  It is owned by
    backend.orchestration.agent_loop.AgentLoop, which calls execute_turn
    once per iteration.  This keeps the core agent a true core agent:
    model-call semantics only, no orchestration.

Entities in it:
    - CoreAgent: One LLM turn with retries for a single agent node.

How used by other modules:
    - backend.orchestration.agent_loop drives CoreAgent.execute_turn once
      per loop iteration.
    - backend.orchestration.executor and backend.orchestration.group
      construct CoreAgent instances from NodeDefinition configurations.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.schema.models import NodeConfig

_LOGGER = logging.getLogger(__name__)


class CoreAgent:
    """
    Performs a single LLM turn for one agent node, with retries.

    Description:
        Wraps the LLM provider call with the node's generation settings
        and retry policy.  ``execute_turn`` sends the given conversation
        once and returns the assistant message — it never loops, never
        dispatches tools, and never decides task completion.  Those are
        orchestration concerns handled by AgentLoop.

    Attributes:
        node_id: Unique identifier for this agent instance.
        label: Human-readable label for this agent.
        config: NodeConfig with model, generation, and retry settings.
        llm_provider: LLMProvider instance for API calls.

    Methods:
        execute_turn: Run one LLM turn and return the assistant message.
    """

    def __init__(
        self,
        node_id: str,
        label: str,
        config: NodeConfig,
        llm_provider: LLMProvider,
        emit_event: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the CoreAgent with its configuration and provider.

        Description:
            Stores the identity, node configuration, provider, and trace
            callback needed to perform LLM turns.

        Params:
            node_id (str): Unique agent node identifier.
            label (str): Human-readable agent label.
            config (NodeConfig): Agent configuration from the workflow schema.
                Consumed fields: response_format, tool_choice,
                parallel_tool_calls, retries, retry_waiting_time.
            llm_provider (LLMProvider): Provider for LLM API calls.
            emit_event (Optional[Callable]): Optional async-compatible callback
                receiving event dicts for tracing (llm_retry events).
                Signature: async (event: dict) -> None

        Returns:
            None
        """
        self.node_id = node_id
        self.label = label
        self.config = config
        self.llm_provider = llm_provider
        self._emit_event = emit_event

    async def execute_turn(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Run one LLM request–response turn.

        Description:
            Sends *messages* to the provider with the node's generation
            settings.  Attempts the call up to ``config.retries + 1``
            times with exponential backoff (base ``retry_waiting_time``);
            every retry is traced via an ``llm_retry`` event.  When all
            attempts fail, the last provider error is re-raised so the
            caller sees the exact failure — there is no fallback value.

        Params:
            messages (list[dict]): Conversation messages to send.
            tools (Optional[list[dict]]): Tool definitions available to the
                LLM, or None.  tool_choice and parallel_tool_calls are only
                forwarded when tools are present.

        Returns:
            dict: Result of the turn with keys:
                - assistant_message (dict): The full assistant message
                  (content and/or tool_calls).
                - finish_reason (str): The provider's finish reason.

        Raises:
            LLMProviderError: The last provider error, when all
                ``config.retries + 1`` attempts failed.
        """
        tool_choice = self.config.tool_choice if tools else None
        parallel_tool_calls = self.config.parallel_tool_calls if tools else None
        max_attempts = self.config.retries + 1
        last_error: Optional[LLMProviderError] = None

        for attempt_index in range(max_attempts):
            try:
                response = await self.llm_provider.complete(
                    messages=messages,
                    tools=tools,
                    response_format=self.config.response_format,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                choice = response["choices"][0]
                return {
                    "assistant_message": choice["message"],
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            except LLMProviderError as provider_error:
                last_error = provider_error
                if attempt_index < self.config.retries:
                    backoff_seconds = self.config.retry_waiting_time ** attempt_index
                    _LOGGER.warning(
                        "[%s] LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        self.node_id, attempt_index + 1, max_attempts,
                        provider_error, backoff_seconds,
                    )
                    await self._emit({
                        "type": "llm_retry",
                        "attempt": attempt_index + 1,
                        "max_attempts": max_attempts,
                        "error": str(provider_error),
                        "backoff_seconds": backoff_seconds,
                    })
                    await asyncio.sleep(backoff_seconds)

        _LOGGER.error("[%s] All %d LLM call attempts failed", self.node_id, max_attempts)
        raise last_error

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
        event["node_id"] = self.node_id
        result = self._emit_event(event)
        if asyncio.iscoroutine(result):
            await result
