"""
Execution harness for tool-call routing, authorization, and rate limiting.

What it does:
    Receives the full LLM assistant message, extracts one ToolRequest per
    entry in its structured ``tool_calls`` block, and dispatches each to
    the tool registry.  Gates every outbound tool call through
    authorization, budget, and rate-limit checks. Handles artifact writing
    for large results.  A message without ``tool_calls`` is a final
    answer: the harness returns no results and never re-interprets text
    as a tool request.

Entities in it:
    - ToolRequest: normalized structure the harness produces from LLM output.
    - ToolAuthorizationError: raised when a tool is not in the authorized set.
    - ToolBudgetExhaustedError: raised when the call budget is exhausted.
    - ExecutionHarness: the stateful gate that wraps every tool invocation.

How used by other modules:
    The orchestration executor creates one ExecutionHarness per workflow node.
    The AgentLoop calls ``process_response(assistant_message)`` (wired in as
    its tool_call_handler) which extracts the ToolRequest structure,
    dispatches to tools, handles artifacts, and returns formatted messages
    ready for the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import copy

from backend.tools.artifact import ArtifactStore
from backend.tools.base import ToolResult
from backend.tools.registry import ToolRegistry
from backend.settings.models import UserSettings

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strict-mode schema transform
# ---------------------------------------------------------------------------

def _make_strict_schema(schema: dict) -> dict:
    """Transform a JSON Schema so it satisfies OpenAI strict-mode constraints.

    Requirements per the OpenAI function-calling spec when ``strict: true``:
    - Every ``object`` node must have ``additionalProperties: false``.
    - ALL properties must appear in ``required``; optional fields use a
      nullable type union (e.g. ``["string", "null"]``) instead of being
      absent from ``required``.
    """
    out = copy.deepcopy(schema)
    _enforce_strict(out)
    return out


def _enforce_strict(node: dict) -> None:
    """Recursively enforce strict-mode constraints on *node* in place."""
    node_type = node.get("type")

    if node_type == "object" and "properties" in node:
        node["additionalProperties"] = False
        existing_required = set(node.get("required", []))
        all_props = list(node["properties"].keys())

        for prop_name in all_props:
            prop = node["properties"][prop_name]
            if prop_name not in existing_required:
                ptype = prop.get("type")
                if ptype is not None and ptype != "null":
                    if isinstance(ptype, list):
                        if "null" not in ptype:
                            prop["type"] = ptype + ["null"]
                    else:
                        prop["type"] = [ptype, "null"]
            _enforce_strict(prop)

        node["required"] = all_props

    elif node_type == "array" and "items" in node:
        _enforce_strict(node["items"])


# ---------------------------------------------------------------------------
# ToolRequest — the structure the harness produces from an LLM response
# ---------------------------------------------------------------------------

@dataclass
class ToolRequest:
    """Normalized tool request extracted from an LLM response.

    One entry per structured tool call in the assistant message's
    ``tool_calls`` block.  The tool_calls block is the single
    authoritative signal that the agent requested a tool — plain text
    content is never re-interpreted as a tool request.
    """
    tool_call_id: str
    tool_name: str
    arguments: dict


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
    """Produces ToolRequests from LLM output, dispatches to tools, handles artifacts.

    The harness extracts one ToolRequest per entry in the assistant
    message's structured ``tool_calls`` block and dispatches each through
    authorization, budget, and rate-limit gates to the tool registry.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        user_settings: UserSettings,
        authorized_tools: set[str],
        call_budget: int,
        rate_limit_per_minute: int,
        run_id: str = "",
        event_callback: Any | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.user_settings = user_settings
        self.authorized_tools = frozenset(authorized_tools)
        self.call_budget = call_budget
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count: int = 0
        self._call_timestamps: list[float] = []
        self._artifact_store = ArtifactStore(run_id) if run_id else None
        self._event_callback = event_callback
        # One entry per completed dispatch: {tool_name, arguments, data_type,
        # content}.  The executor reads this to route results to TOOL nodes.
        self.calls_log: list[dict[str, Any]] = []

    # -- primary interface: called by the AgentLoop ---------------------------

    async def process_response(self, assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a full LLM assistant message through the tool pipeline.

        1. Extract one ToolRequest per entry in the ``tool_calls`` block
        2. For each request, dispatch through the authorization/budget gates
        3. Handle artifacts for large results
        4. Return formatted messages for the conversation

        Returns:
            List of message dicts to append to conversation. Empty if the
            message contains no structured tool_calls.
        """
        tool_calls_block = assistant_message.get("tool_calls")
        if not tool_calls_block:
            return []

        results = []
        for tool_call in tool_calls_block:
            request = ToolRequest(
                tool_call_id=tool_call["id"],
                tool_name=tool_call["function"]["name"],
                arguments=json.loads(tool_call["function"].get("arguments", "{}")),
            )
            entry = await self._dispatch(request)
            results.append(
                self._format_result_message(entry, tool_call_id=request.tool_call_id)
            )
        return results

    def get_tool_definitions(
        self, tool_names: list[str], *, strict: bool = True,
    ) -> list[dict[str, Any]]:
        """Build OpenAI function-calling descriptors for the given tools.

        When *strict* is True, each function definition includes
        ``"strict": true`` and the parameters schema is transformed to
        meet the OpenAI structured-output requirements (all properties
        required, optional ones made nullable, ``additionalProperties``
        set to false on every object node).
        """
        definitions: list[dict[str, Any]] = []
        for name in tool_names:
            tool = self.tool_registry.get(name)
            schema = tool.parameters_schema
            if strict:
                schema = _make_strict_schema(schema)
            defn: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            }
            if strict:
                defn["function"]["strict"] = True
            definitions.append(defn)
        return definitions

    def reset_budget(self) -> None:
        """Reset the call counter and sliding-window timestamps to zero."""
        self._call_count = 0
        self._call_timestamps.clear()

    # -- dispatch logic -----------------------------------------------------

    async def _dispatch(self, request: ToolRequest) -> dict[str, Any]:
        """Route a ToolRequest through the gates to the tool registry.

        Enforces authorization, call budget, and rate limit, then executes
        the tool with the request's arguments.

        Returns:
            The normalized call-log entry (also appended to ``calls_log``).

        Raises:
            ToolAuthorizationError: If the tool is outside the authorized set.
            ToolBudgetExhaustedError: If the call budget is spent.
        """
        tool_name = request.tool_name
        arguments = request.arguments

        if tool_name not in self.authorized_tools:
            raise ToolAuthorizationError(tool_name, set(self.authorized_tools))
        if self._call_count >= self.call_budget:
            raise ToolBudgetExhaustedError(self.call_budget, self._call_count)
        await self._enforce_rate_limit()

        tool = self.tool_registry.get(tool_name)
        self._inject_credentials(tool)
        tool.inject_event_callback(self._event_callback)

        await self._emit_tool_event("tool_call", tool_name, arguments)

        _LOGGER.info("Tool dispatch: %s(%s)", tool_name, list(arguments.keys()))
        result = await tool.execute(**arguments)
        return await self._finalize_call(tool, tool_name, arguments, result)

    async def _finalize_call(
        self,
        tool: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        """Single convergence point after a tool executes.

        Counts the call, validates the output against the tool's
        declared schema, emits the tool_result event, stringifies the
        content (artifact-aware), and appends the normalized entry to
        ``calls_log``.
        """
        self._call_count += 1

        output_schema = getattr(tool, "output_schema", None)
        if output_schema is not None:
            self._validate_tool_output(tool_name, result, output_schema)

        tool_result = result if isinstance(result, ToolResult) else ToolResult(data_type="raw", content=result)
        await self._emit_tool_event("tool_result", tool_name, None, tool_result)

        entry = {
            "tool_name": tool_name,
            "arguments": arguments,
            "data_type": tool_result.data_type,
            "content": self._result_content_str(tool_result),
        }
        self.calls_log.append(entry)
        return entry

    async def _emit_tool_event(
        self,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any] | None,
        tool_result: ToolResult | None = None,
    ) -> None:
        """Emit a tool_call or tool_result event via the injected callback."""
        if self._event_callback is None:
            return
        event: dict[str, Any] = {"type": event_type, "tool_name": tool_name}
        if event_type == "tool_call" and arguments is not None:
            event["arguments"] = arguments
        elif event_type == "tool_result" and tool_result is not None:
            content = tool_result.content
            preview = content[:2000] if isinstance(content, str) else str(content)[:2000]
            event["data_type"] = tool_result.data_type
            event["content_preview"] = preview
        cb_result = self._event_callback(event)
        if asyncio.iscoroutine(cb_result):
            await cb_result

    # -- result formatting with artifact handling ---------------------------

    def _result_content_str(self, tool_result: ToolResult) -> str:
        """Stringify a ToolResult's content, artifact-aware.

        Small results are inlined. Large results are written once to the
        artifact store and replaced by a pointer.
        """
        if self._artifact_store and self._artifact_store.is_large(tool_result.content):
            artifact_path = self._artifact_store.write(tool_result.data_type, tool_result.content)
            return (
                f"[{tool_result.data_type} data extracted. "
                f"Available at {artifact_path}. "
                f"Access this artifact when needed for analysis.]"
            )
        if isinstance(tool_result.content, str):
            return tool_result.content
        return json.dumps(tool_result.content, default=str)

    def _format_result_message(self, entry: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        """Format a call-log entry into a tool-role conversation message."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": entry["content"]}

    # -- helpers ------------------------------------------------------------

    def _inject_credentials(self, tool: Any) -> None:
        """Inject user credentials and enabled-source config into a tool instance."""
        credentials: dict[str, Any] = {}
        for api_cred in self.user_settings.api_credentials:
            credentials[api_cred.credential_name] = api_cred.fields
        credentials["_enabled_public_sources"] = list(
            self.user_settings.enabled_public_sources
        )
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
