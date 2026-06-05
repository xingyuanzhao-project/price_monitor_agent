"""
Harness subpackage for context assembly and tool execution control.

What it does:
    Provides two complementary harnesses that sit between the orchestration
    layer and the core agent.  The ContextHarness assembles LLM message
    sequences with token-budget awareness, few-shot examples, and guardrail
    enforcement.  The ExecutionHarness routes tool calls through authorization,
    budget, and rate-limit gates before delegating to the tool registry.

Entities in it:
    - context: ContextHarness, GuardrailViolationError, GuardrailRule.
    - execution: ExecutionHarness, ToolAuthorizationError,
      ToolBudgetExhaustedError.

How used by other modules:
    The orchestration executor instantiates both harnesses per workflow node
    and passes them into CoreAgent / AgentGroup.  The API layer never touches
    harnesses directly.
"""

from backend.harness.context import (
    ContextHarness,
    GuardrailRule,
    GuardrailViolationError,
)
from backend.harness.execution import (
    ExecutionHarness,
    ToolAuthorizationError,
    ToolBudgetExhaustedError,
)
