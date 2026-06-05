"""
LLM provider abstraction for async communication with language model APIs.

What it does:
    Provides an async HTTP client wrapper for OpenRouter-compatible LLM APIs.
    Supports standard completions, streaming completions, and model listing.
    Propagates all errors with full context (status code and response body).

Entities in it:
    - LLMProviderError: Exception raised on any LLM API communication failure.
    - LLMProvider: Async client for LLM API interactions.

How used by other modules:
    - backend.agent.core instantiates LLMProvider and calls complete() during
      the agentic loop for non-streaming execution.
    - backend.agent.group uses LLMProvider for the planner phase to decompose
      tasks into sub-agent assignments.
    - The orchestration engine calls list_models() to populate available models.
    - close() is called during graceful shutdown to release HTTP connections.
"""

from typing import AsyncIterator, Optional

import httpx


class LLMProviderError(Exception):
    """
    Raised when an LLM API call fails.

    Description:
        Carries the HTTP status code (if available) and the response body
        text to provide full diagnostic context for the failure.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code from the API response, or None.
        response_body: Raw response body text, or None.
    """

    def __init__(
        self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None
    ) -> None:
        """
        Initialize with error details.

        Description:
            Stores the status code and response body alongside the message.

        Params:
            message (str): Human-readable error description.
            status_code (Optional[int]): HTTP status code if available.
            response_body (Optional[str]): Raw response body if available.

        Returns:
            None
        """
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
    """
    Async client for communicating with OpenRouter-compatible LLM APIs.

    Description:
        Manages an HTTP client session for making chat completion requests,
        streaming responses, and listing available models. All errors are
        raised as LLMProviderError with full diagnostic context.

    Attributes:
        api_key: Authentication key for the LLM API.
        model_id: Default model identifier for completions.
        temperature: Default sampling temperature.
        max_tokens: Optional default maximum response tokens.
        base_url: API base URL endpoint.
        _client: Internal httpx.AsyncClient instance.

    Methods:
        complete: Make a non-streaming chat completion request.
        complete_stream: Make a streaming chat completion request.
        list_models: Retrieve available models from the API.
        close: Close the HTTP client session.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        """
        Initialize the LLM provider with connection parameters.

        Description:
            Stores configuration and creates an async HTTP client with
            appropriate headers and timeout settings.

        Params:
            api_key (str): Authentication key for the API.
            model_id (str): Default model identifier.
            temperature (float): Default sampling temperature.
            max_tokens (Optional[int]): Default max response tokens.
            base_url (str): API base URL.

        Returns:
            None
        """
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        """
        Make a non-streaming chat completion request.

        Description:
            Sends messages to the LLM API and returns the complete response.
            Optionally includes tool definitions and response format constraints.

        Params:
            messages (list[dict]): Chat messages in OpenAI format.
            tools (Optional[list[dict]]): Tool definitions for function calling.
            response_format (Optional[dict]): Structured output format specification.

        Returns:
            dict: The API response containing choices with message content and
                  optional tool_calls.

        Raises:
            LLMProviderError: On any API communication failure.
        """
        request_body = self._build_request_body(messages, tools, response_format, stream=False)

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=request_body,
            )
        except httpx.HTTPError as http_error:
            raise LLMProviderError(
                f"HTTP request to LLM API failed: {http_error}",
                status_code=None,
                response_body=None,
            ) from http_error

        if response.status_code != 200:
            raise LLMProviderError(
                f"LLM API returned non-200 status for model '{self.model_id}'",
                status_code=response.status_code,
                response_body=response.text,
            )

        try:
            return response.json()
        except ValueError as parse_error:
            raise LLMProviderError(
                f"Failed to parse LLM API response as JSON: {parse_error}",
                status_code=response.status_code,
                response_body=response.text,
            ) from parse_error

    async def complete_stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
    ) -> AsyncIterator[dict]:
        """
        Make a streaming chat completion request.

        Description:
            Sends messages to the LLM API with streaming enabled and yields
            parsed server-sent event chunks as they arrive.

        Params:
            messages (list[dict]): Chat messages in OpenAI format.
            tools (Optional[list[dict]]): Tool definitions for function calling.
            response_format (Optional[dict]): Structured output format specification.

        Returns:
            AsyncIterator[dict]: Iterator yielding parsed SSE chunk dictionaries.

        Raises:
            LLMProviderError: On any API communication failure.
        """
        request_body = self._build_request_body(messages, tools, response_format, stream=True)

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise LLMProviderError(
                        f"LLM API streaming returned non-200 status for model '{self.model_id}'",
                        status_code=response.status_code,
                        response_body=error_body.decode("utf-8", errors="replace"),
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_content = line[6:]
                    if data_content.strip() == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data_content)
                        yield chunk
                    except ValueError:
                        continue

        except httpx.HTTPError as http_error:
            raise LLMProviderError(
                f"HTTP streaming request to LLM API failed: {http_error}",
                status_code=None,
                response_body=None,
            ) from http_error

    async def list_models(self) -> list[str]:
        """
        Retrieve available model identifiers from the API.

        Description:
            Queries the models endpoint and returns a list of model ID strings.

        Params:
            None

        Returns:
            list[str]: List of available model identifier strings.

        Raises:
            LLMProviderError: On any API communication failure.
        """
        try:
            response = await self._client.get(f"{self.base_url}/models")
        except httpx.HTTPError as http_error:
            raise LLMProviderError(
                f"Failed to list models from LLM API: {http_error}",
                status_code=None,
                response_body=None,
            ) from http_error

        if response.status_code != 200:
            raise LLMProviderError(
                "LLM API models endpoint returned non-200 status",
                status_code=response.status_code,
                response_body=response.text,
            )

        try:
            response_data = response.json()
        except ValueError as parse_error:
            raise LLMProviderError(
                f"Failed to parse models response as JSON: {parse_error}",
                status_code=response.status_code,
                response_body=response.text,
            ) from parse_error

        models_list = response_data.get("data", [])
        return [model["id"] for model in models_list if "id" in model]

    async def close(self) -> None:
        """
        Close the HTTP client session and release resources.

        Description:
            Gracefully shuts down the underlying httpx async client.

        Params:
            None

        Returns:
            None
        """
        await self._client.aclose()

    def _build_request_body(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        response_format: Optional[dict],
        stream: bool,
    ) -> dict:
        """
        Build the request body dictionary for a chat completion call.

        Description:
            Constructs the JSON request body with model, messages, temperature,
            and optional parameters (max_tokens, tools, response_format, stream).

        Params:
            messages (list[dict]): Chat messages.
            tools (Optional[list[dict]]): Tool definitions.
            response_format (Optional[dict]): Output format spec.
            stream (bool): Whether to enable streaming.

        Returns:
            dict: Complete request body dictionary.
        """
        body: dict = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        if tools:
            body["tools"] = tools

        if response_format:
            body["response_format"] = response_format

        return body
