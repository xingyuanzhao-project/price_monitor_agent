"""
Execution harness for tool-call routing, authorization, and rate limiting.

What it does:
    Gates every outbound tool call through three sequential checks —
    authorization (is the tool in the node's allowed set?), budget
    (has the call ceiling been reached?), and rate-limit (sliding-window
    60-second cap) — before delegating to the tool registry.  Also
    injects API credentials from user settings when a tool declares a
    ``credential_name``.

Entities in it:
    - ToolAuthorizationError: raised when a tool is not in the authorized set.
    - ToolBudgetExhaustedError: raised when the call budget is exhausted.
    - ExecutionHarness: the stateful gate that wraps every tool invocation.

How used by other modules:
    The orchestration executor creates one ExecutionHarness per workflow node,
    injecting the node's authorized tool set and budget.  The harness is then
    passed into CoreAgent / AgentGroup which call ``handle_tool_call`` during
    the agentic loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from backend.tools.registry import ToolRegistry
from backend.settings.models import UserSettings


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ToolAuthorizationError(Exception):
    """Raised when a tool call targets a tool outside the authorized set.

    Attributes:
        tool_name: The requested tool.
        authorized_tools: The set of tools this harness is permitted to call.
    """

    def __init__(self, tool_name: str, authorized_tools: set[str]) -> None:
        self.tool_name = tool_name
        self.authorized_tools = authorized_tools
        super().__init__(
            f"Tool '{tool_name}' is not authorized. "
            f"Authorized tools: {sorted(authorized_tools)}"
        )


class ToolBudgetExhaustedError(Exception):
    """Raised when all permitted tool calls have been consumed.

    Attributes:
        call_budget: The configured ceiling.
        call_count: How many calls were made before exhaustion.
    """

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
    """Stateful gate for tool-call routing with auth, budget, and rate control.

    Attributes:
        tool_registry: Central registry from which tools are retrieved.
        user_settings: Settings object used for credential look-up.
        authorized_tools: Names of tools this harness may invoke.
        call_budget: Maximum number of tool calls permitted.
        rate_limit_per_minute: Maximum calls in any 60-second sliding window.

    Methods:
        handle_tool_call: Execute a single tool call through all gates.
        get_tool_definitions: Return OpenAI function-calling descriptors.
        reset_budget: Zero out call counts and timestamps.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        user_settings: UserSettings,
        authorized_tools: set[str],
        call_budget: int,
        rate_limit_per_minute: int,
    ) -> None:
        """Initialise the execution harness.

        Args:
            tool_registry: Registry of available tool instances.
            user_settings: User settings (credentials, providers).
            authorized_tools: Set of tool names allowed for this node.
            call_budget: Maximum total tool calls.
            rate_limit_per_minute: Maximum calls per 60-second window.
        """
        self.tool_registry = tool_registry
        self.user_settings = user_settings
        self.authorized_tools = frozenset(authorized_tools)
        self.call_budget = call_budget
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count: int = 0
        self._call_timestamps: list[float] = []

    # -- public API ---------------------------------------------------------

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Route a tool call through authorization, budget, and rate checks.

        Sequence: auth → budget → rate-limit → registry look-up → credential
        injection → execution → count increment.

        Args:
            tool_name: Registered name of the target tool.
            arguments: Keyword arguments forwarded to ``tool.execute``.

        Returns:
            The tool's return value, unchanged.

        Raises:
            ToolAuthorizationError: If *tool_name* is not authorized.
            ToolBudgetExhaustedError: If the call budget is exhausted.
            KeyError: If the tool is not found in the registry (propagated).
        """
        if tool_name not in self.authorized_tools:
            raise ToolAuthorizationError(tool_name, set(self.authorized_tools))

        if self._call_count >= self.call_budget:
            raise ToolBudgetExhaustedError(self.call_budget, self._call_count)

        await self._enforce_rate_limit()

        tool = self.tool_registry.get(tool_name)

        credentials: dict[str, Any] = {}
        for api_cred in self.user_settings.api_credentials:
            credentials[api_cred.credential_name] = api_cred.fields
        tool.inject_credentials(credentials)

        result = await tool.execute(**arguments)

        self._call_count += 1

        return result

    def get_tool_definitions(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """Build OpenAI function-calling descriptors for the given tools.

        Args:
            tool_names: Tool names to include in the definition list.

        Returns:
            A list of dicts in OpenAI ``tools`` format, each containing
            ``type`` (``"function"``) and a ``function`` dict with ``name``,
            ``description``, and ``parameters``.

        Raises:
            KeyError: If a tool name is not found in the registry.
        """
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

    # -- private helpers ----------------------------------------------------

    async def _enforce_rate_limit(self) -> None:
        """Block until a rate-limit slot is available (sliding 60-s window).

        If the window is full, sleeps until the oldest call expires rather
        than raising an error.
        """
        now = time.monotonic()
        self._call_timestamps = [
            ts for ts in self._call_timestamps if now - ts < 60.0
        ]

        if len(self._call_timestamps) >= self.rate_limit_per_minute:
            wait_seconds = 60.0 - (now - self._call_timestamps[0])
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            now = time.monotonic()
            self._call_timestamps = [
                ts for ts in self._call_timestamps if now - ts < 60.0
            ]

        self._call_timestamps.append(time.monotonic())
