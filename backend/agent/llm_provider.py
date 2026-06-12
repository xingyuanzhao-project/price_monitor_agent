"""
LLM provider abstraction using the OpenAI SDK for async communication.

What it does:
    Wraps ``openai.AsyncOpenAI`` to provide a uniform interface for all
    providers (cloud and local) that expose an OpenAI-compatible API.
    Handles provider-specific format normalization via format_adapter.

Entities in it:
    - LLMProviderError: Exception raised on any LLM API communication failure.
    - LLMProvider: Async client for LLM API interactions via OpenAI SDK.

How used by other modules:
    - backend.agent.core calls complete() once per LLM turn (CoreAgent is
      driven by backend.orchestration.agent_loop.AgentLoop).
    - backend.orchestration.group uses LLMProvider for the planner phase to
      decompose tasks into sub-agent assignments.
    - The orchestration engine calls list_models() to populate available models.
    - close() is called during graceful shutdown to release connections.
"""

import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError

from backend.agent.format_adapter import normalize_request_kwargs

_LOGGER = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM API call fails.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code from the API response, or None.
        response_body: Raw response body text, or None.
    """

    def __init__(
        self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        full_message = message
        if status_code is not None:
            full_message += f" [HTTP {status_code}]"
        if response_body is not None:
            full_message += f" Body: {response_body[:500]}"
        super().__init__(full_message)


class LLMProvider:
    """Async client for communicating with OpenAI-compatible LLM APIs.

    Uses the ``openai.AsyncOpenAI`` SDK which provides built-in retry logic,
    streaming support, and proper error handling across all compatible providers.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        provider: str = "openrouter",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=max_retries,
        )

    async def complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> dict:
        """Make a non-streaming chat completion request.

        Returns:
            dict: The API response as a dictionary with choices.
        """
        kwargs = self._build_kwargs(
            messages, tools, response_format, stream=False,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
        )
        kwargs = normalize_request_kwargs(self.provider, kwargs)
        _LOGGER.info("LLM request → provider=%s model=%s messages=%d tools=%d",
                     self.provider, self.model_id, len(messages),
                     len(tools) if tools else 0)
        if messages:
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user:
                content_preview = (last_user.get("content") or "")[:200]
                _LOGGER.info("LLM prompt (user): %s", content_preview)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except (APIError, APIConnectionError, APITimeoutError) as api_error:
            status = getattr(api_error, "status_code", None)
            body = getattr(api_error, "body", None)
            _LOGGER.error("LLM API error: model=%s status=%s error=%s",
                          self.model_id, status, api_error)
            raise LLMProviderError(
                f"LLM API call failed for model '{self.model_id}': {api_error}",
                status_code=status,
                response_body=str(body) if body else None,
            ) from api_error

        result = response.model_dump()
        choice = result.get("choices", [{}])[0] if result.get("choices") else {}
        content = (choice.get("message") or {}).get("content") or ""
        _LOGGER.info("LLM response ← model=%s finish=%s content_length=%d",
                     self.model_id,
                     choice.get("finish_reason", "?"),
                     len(content))
        return result

    async def complete_stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> AsyncIterator[dict]:
        """Make a streaming chat completion request.

        Yields:
            dict: Parsed SSE chunk dictionaries.
        """
        kwargs = self._build_kwargs(
            messages, tools, response_format, stream=True,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
        )
        kwargs = normalize_request_kwargs(self.provider, kwargs)

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                yield chunk.model_dump()
        except (APIError, APIConnectionError, APITimeoutError) as api_error:
            status = getattr(api_error, "status_code", None)
            body = getattr(api_error, "body", None)
            raise LLMProviderError(
                f"LLM API streaming failed for model '{self.model_id}': {api_error}",
                status_code=status,
                response_body=str(body) if body else None,
            ) from api_error

    async def list_models(self) -> list[str]:
        """Retrieve available model identifiers from the API."""
        try:
            models_page = await self._client.models.list()
            return [model.id for model in models_page.data]
        except (APIError, APIConnectionError, APITimeoutError) as api_error:
            raise LLMProviderError(
                f"Failed to list models: {api_error}",
                status_code=getattr(api_error, "status_code", None),
                response_body=None,
            ) from api_error

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self._client.close()

    def _build_kwargs(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        response_format: Optional[dict],
        stream: bool,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> dict:
        """Build keyword arguments for the chat completions create call."""
        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = parallel_tool_calls

        if response_format:
            kwargs["response_format"] = response_format

        return kwargs
