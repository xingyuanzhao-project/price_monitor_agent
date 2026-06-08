"""
Execution harness for tool-call routing, authorization, and rate limiting.

What it does:
    Receives the full LLM assistant message and produces the normalized
    ToolRequest structure: {native_tool_request, non_native_tool_request}.
    Then dispatches to the tool support layer, which decides how to handle
    each path. Gates every outbound tool call through authorization, budget,
    and rate-limit checks. Handles artifact writing for large results.

Entities in it:
    - ToolRequest: normalized structure the harness produces from LLM output.
    - ToolAuthorizationError: raised when a tool is not in the authorized set.
    - ToolBudgetExhaustedError: raised when the call budget is exhausted.
    - ExecutionHarness: the stateful gate that wraps every tool invocation.

How used by other modules:
    The orchestration executor creates one ExecutionHarness per workflow node.
    CoreAgent calls ``process_response(assistant_message)`` which extracts the
    ToolRequest structure, dispatches to tools, handles artifacts, and returns
    formatted messages ready for the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.tools.artifact import ArtifactStore
from backend.tools.base import ToolResult
from backend.tools.registry import ToolRegistry
from backend.settings.models import UserSettings

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolRequest — the structure the harness produces from an LLM response
# ---------------------------------------------------------------------------

@dataclass
class ToolRequest:
    """Normalized tool request extracted from an LLM response.

    The harness always produces this same structure. One field is populated,
    the other is None. The tool support layer decides what to do with it.
    """
    native_tool_request: dict | None
    non_native_tool_request: str | None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ToolOutputValidationError(Exception):
    """Raised when a tool's return value does not match its declared output schema."""

    def __init__(self, tool_name: str, validation_message: str) -> None:
        self.tool_name = tool_name
        self.validation_message = validation_message
        super().__init__(
            f"Tool '{tool_name}' output validation failed: {validation_message}"
        )


class ToolAuthorizationError(Exception):
    """Raised when a tool call targets a tool outside the authorized set."""

    def __init__(self, tool_name: str, authorized_tools: set[str]) -> None:
        self.tool_name = tool_name
        self.authorized_tools = authorized_tools
        super().__init__(
            f"Tool '{tool_name}' is not authorized. "
            f"Authorized tools: {sorted(authorized_tools)}"
        )


class ToolBudgetExhaustedError(Exception):
    """Raised when all permitted tool calls have been consumed."""

    def __init__(self, call_budget: int, call_count: int) -> None:
        self.call_budget = call_budget
        self.call_count = call_count
        super().__init__(
            f"Tool call budget exhausted: {call_count}/{call_budget} calls used"
        )


# ---------------------------------------------------------------------------
# Execution harness
# ---------------------------------------------------------------------------

class ExecutionHarness:
    """Produces ToolRequest from LLM output, dispatches to tools, handles artifacts.

    The harness extracts the structure {native_tool_request, non_native_tool_request}
    from the assistant message. The tool support layer receives this and decides:
    - If native_tool_request exists → dispatch directly
    - If non_native_tool_request → tool uses parse_request to extract canonical args
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        user_settings: UserSettings,
        authorized_tools: set[str],
        call_budget: int,
        rate_limit_per_minute: int,
        run_id: str = "",
    ) -> None:
        self.tool_registry = tool_registry
        self.user_settings = user_settings
        self.authorized_tools = frozenset(authorized_tools)
        self.call_budget = call_budget
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count: int = 0
        self._call_timestamps: list[float] = []
        self._artifact_store = ArtifactStore(run_id) if run_id else None

    # -- primary interface: called by CoreAgent ------------------------------

    async def process_response(self, assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a full LLM assistant message through the tool pipeline.

        1. Extract ToolRequest(s) from the message
        2. For each request, dispatch to the tool support layer
        3. Handle artifacts for large results
        4. Return formatted messages for the conversation

        Returns:
            List of message dicts to append to conversation. Empty if no tool
            action was taken (non-native path found no match).
        """
        tool_calls_block = assistant_message.get("tool_calls")

        if tool_calls_block:
            # Native path: LLM produced structured tool_calls
            results = []
            for tc in tool_calls_block:
                request = ToolRequest(
                    native_tool_request={
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"].get("arguments", "{}")),
                    },
                    non_native_tool_request=None,
                )
                tool_result = await self._dispatch(request)
                formatted = self._format_result_message(tool_result, tool_call_id=tc["id"])
                results.append(formatted)
            return results

        content = assistant_message.get("content", "")
        if content:
            # Non-native path: LLM produced text, pass to tool layer for parsing
            request = ToolRequest(
                native_tool_request=None,
                non_native_tool_request=content,
            )
            tool_result = await self._dispatch(request)
            if tool_result is not None:
                formatted = self._format_result_message(tool_result, tool_call_id=None)
                return [formatted]

        return []

    # -- legacy interface (kept for direct TOOL node execution) --------------

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Direct tool dispatch with auth/budget/rate checks. Used for TOOL nodes."""
        if tool_name not in self.authorized_tools:
            raise ToolAuthorizationError(tool_name, set(self.authorized_tools))

        if self._call_count >= self.call_budget:
            raise ToolBudgetExhaustedError(self.call_budget, self._call_count)

        await self._enforce_rate_limit()

        tool = self.tool_registry.get(tool_name)
        self._inject_credentials(tool)

        result = await tool.execute(**arguments)
        self._call_count += 1

        output_schema = getattr(tool, "output_schema", None)
        if output_schema is not None:
            self._validate_tool_output(tool_name, result, output_schema)

        return result

    def get_tool_definitions(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """Build OpenAI function-calling descriptors for the given tools."""
        definitions: list[dict[str, Any]] = []
        for name in tool_names:
            tool = self.tool_registry.get(name)
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            })
        return definitions

    def reset_budget(self) -> None:
        """Reset the call counter and sliding-window timestamps to zero."""
        self._call_count = 0
        self._call_timestamps.clear()

    # -- dispatch logic -----------------------------------------------------

    async def _dispatch(self, request: ToolRequest) -> ToolResult | None:
        """Route a ToolRequest to the tool support layer.

        The tool layer decides:
        - native_tool_request exists → use name + args directly
        - non_native_tool_request → iterate authorized tools, call parse_request,
          first match wins
        """
        if request.native_tool_request:
            tool_name = request.native_tool_request["name"]
            arguments = request.native_tool_request["arguments"]

            if tool_name not in self.authorized_tools:
                raise ToolAuthorizationError(tool_name, set(self.authorized_tools))
            if self._call_count >= self.call_budget:
                raise ToolBudgetExhaustedError(self.call_budget, self._call_count)
            await self._enforce_rate_limit()

            tool = self.tool_registry.get(tool_name)
            self._inject_credentials(tool)

            _LOGGER.info("Native tool dispatch: %s(%s)", tool_name, list(arguments.keys()))
            result = await tool.execute(**arguments)
            self._call_count += 1

            if isinstance(result, ToolResult):
                return result
            return ToolResult(data_type="raw", content=result)

        elif request.non_native_tool_request:
            text = request.non_native_tool_request
            _LOGGER.info("Non-native tool request: trying parse_request on %d authorized tools",
                         len(self.authorized_tools))

            for tool_name in self.authorized_tools:
                if self._call_count >= self.call_budget:
                    break
                try:
                    tool = self.tool_registry.get(tool_name)
                except KeyError:
                    continue

                parsed_args = tool.parse_request(text)
                if parsed_args is not None:
                    await self._enforce_rate_limit()
                    self._inject_credentials(tool)

                    _LOGGER.info("Non-native match: %s parsed args=%s", tool_name, list(parsed_args.keys()))
                    result = await tool.execute(**parsed_args)
                    self._call_count += 1

                    if isinstance(result, ToolResult):
                        return result
                    return ToolResult(data_type="raw", content=result)

            return None

        return None

    # -- result formatting with artifact handling ---------------------------

    def _format_result_message(self, tool_result: ToolResult, tool_call_id: str | None) -> dict[str, Any]:
        """Format a ToolResult into a conversation message.

        Small results are inlined. Large results become artifact pointers.
        """
        if self._artifact_store and self._artifact_store.is_large(tool_result.content):
            artifact_path = self._artifact_store.write(tool_result.data_type, tool_result.content)
            content_str = (
                f"[{tool_result.data_type} data extracted. "
                f"Available at {artifact_path}. "
                f"Access this artifact when needed for analysis.]"
            )
        else:
            if isinstance(tool_result.content, str):
                content_str = tool_result.content
            else:
                content_str = json.dumps(tool_result.content, default=str)

        if tool_call_id:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content_str}
        else:
            return {"role": "user", "content": f"[Data retrieved — {tool_result.data_type}]:\n{content_str}"}

    # -- helpers ------------------------------------------------------------

    def _inject_credentials(self, tool: Any) -> None:
        """Inject all user credentials into a tool instance."""
        credentials: dict[str, Any] = {}
        for api_cred in self.user_settings.api_credentials:
            credentials[api_cred.credential_name] = api_cred.fields
        tool.inject_credentials(credentials)

    def _validate_tool_output(
        self, tool_name: str, output: Any, output_schema: dict[str, Any]
    ) -> None:
        """Validate tool output against declared schema."""
        schema_type = output_schema.get("type")
        if schema_type == "object":
            if not isinstance(output, dict):
                raise ToolOutputValidationError(
                    tool_name, f"expected dict but got {type(output).__name__}")
            required_keys = output_schema.get("required", [])
            missing = [k for k in required_keys if k not in output]
            if missing:
                raise ToolOutputValidationError(
                    tool_name, f"missing required keys: {missing}")
        elif schema_type == "array" and not isinstance(output, list):
            raise ToolOutputValidationError(
                tool_name, f"expected list but got {type(output).__name__}")
        elif schema_type == "string" and not isinstance(output, str):
            raise ToolOutputValidationError(
                tool_name, f"expected string but got {type(output).__name__}")

    async def _enforce_rate_limit(self) -> None:
        """Block until a rate-limit slot is available (sliding 60-s window)."""
        now = time.monotonic()
        self._call_timestamps = [ts for ts in self._call_timestamps if now - ts < 60.0]

        if len(self._call_timestamps) >= self.rate_limit_per_minute:
            wait_seconds = 60.0 - (now - self._call_timestamps[0])
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            now = time.monotonic()
            self._call_timestamps = [ts for ts in self._call_timestamps if now - ts < 60.0]

        self._call_timestamps.append(time.monotonic())
