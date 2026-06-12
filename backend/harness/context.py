"""
Context harness for LLM prompt assembly and guardrail enforcement.

What it does:
    Assembles the full message list that the core agent sends to an LLM,
    honouring a token budget, injecting few-shot examples, and enforcing
    input / output guardrail rules.  Upstream data is truncated (largest
    item first) when the budget is tight; system prompts and the user task
    are never truncated.

Entities in it:
    - GuardrailViolationError: raised when any guardrail rule is violated.
    - GuardrailRule: parsed representation of a single guardrail string.
    - ContextHarness: stateful prompt assembler with guardrail validation.

How used by other modules:
    The orchestration executor creates one ContextHarness per workflow node,
    calls ``assemble_messages`` to build the LLM input, then calls
    ``validate_input`` / ``validate_output`` around agent execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GuardrailViolationError(Exception):
    """Raised when one or more guardrail rules are violated.

    Attributes:
        violations: Human-readable descriptions of every triggered rule.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Guardrail violations: {violations}")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardrailRule:
    """Parsed representation of a single guardrail rule.

    Attributes:
        scope: ``"input"`` or ``"output"``.
        rule_type: Kind of check (e.g. ``"max_length"``, ``"forbidden_topic"``).
        value: Threshold, pattern, or phrase used by the check.
    """

    scope: str
    rule_type: str
    value: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_INPUT_TYPES = frozenset({"max_length", "forbidden_topic", "required_pattern"})
_VALID_OUTPUT_TYPES = frozenset({"max_length", "forbidden_phrase", "required_structure"})
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Context harness
# ---------------------------------------------------------------------------

class ContextHarness:
    """Prompt assembler with token-budget management and guardrail enforcement.

    Attributes:
        system_prompt: Developer-level system instruction (empty for regular nodes).
        instruction: User-created instruction prompts sent as user-role messages.
        token_budget: Maximum number of tokens for the assembled message list.
        scope_window: Maximum number of few-shot examples to retain.
        few_shot_examples: Example user/assistant pairs for in-context learning.

    Methods:
        assemble_messages: Build the full LLM message list.
        validate_input: Enforce input guardrail rules (raises on violation).
        validate_output: Enforce output guardrail rules (raises on violation).
    """

    def __init__(
        self,
        system_prompt: str,
        instruction: list[str],
        token_budget: int,
        scope_window: int,
        guardrail_rules: list[str],
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> None:
        """Initialise the context harness.

        Args:
            system_prompt: Developer-level system instruction (empty for regular nodes).
            instruction: User-created instruction prompts from the node config.
            token_budget: Token ceiling (1 token ≈ 4 chars).
            scope_window: How many recent few-shot examples to keep.
            guardrail_rules: Raw strings in ``"scope:rule_type:value"`` format.
            few_shot_examples: Optional list of ``{"user": …, "assistant": …}``
                dicts.

        Raises:
            ValueError: If any guardrail string is malformed.
        """
        self.system_prompt = system_prompt
        self.instruction = list(instruction)
        self.token_budget = token_budget
        self.scope_window = scope_window
        self.few_shot_examples = list(few_shot_examples) if few_shot_examples else []
        self._parsed_rules = _parse_guardrail_rules(guardrail_rules)

    # -- public API ---------------------------------------------------------

    def assemble_messages(
        self,
        user_task: str,
        upstream_data: dict[str, str],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the ordered message list for an LLM call.

        Args:
            user_task: The primary user instruction for this node.
            upstream_data: Mapping of source-node-id → text payload.
            tool_results: Prior tool-call results (OpenAI tool-message dicts).

        Returns:
            A list of message dicts (``role`` / ``content`` / optional
            ``tool_call_id``) ready for the LLM provider.
        """
        messages: list[dict[str, Any]] = []

        # 1. System message (developer-level, only if explicitly set)
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 2. User instruction prompt (from node config "Prompts" field)
        if self.instruction:
            instruction_text = "\n".join(f"- {item}" for item in self.instruction)
            messages.append({"role": "user", "content": instruction_text})

        # 3. Few-shot examples (windowed, never truncated)
        windowed_examples = self.few_shot_examples[-self.scope_window :]
        for example in windowed_examples:
            if "user" in example:
                messages.append({"role": "user", "content": example["user"]})
            if "assistant" in example:
                messages.append({"role": "assistant", "content": example["assistant"]})

        # 4. Compute fixed-cost tokens (system + instruction + examples + tool results + task)
        fixed_tokens = _estimate_tokens(self.system_prompt) + _estimate_tokens(user_task)
        if self.instruction:
            fixed_tokens += _estimate_tokens("\n".join(self.instruction))
        for ex in windowed_examples:
            fixed_tokens += _estimate_tokens(ex.get("user", ""))
            fixed_tokens += _estimate_tokens(ex.get("assistant", ""))
        for tr in tool_results:
            fixed_tokens += _estimate_tokens(str(tr.get("content", "")))

        remaining_budget = max(0, self.token_budget - fixed_tokens)

        # 5. Truncate upstream data (largest first) to fit remaining budget
        if upstream_data:
            data_items = [(key, str(val)) for key, val in upstream_data.items()]
            total_data_tokens = sum(_estimate_tokens(v) for _, v in data_items)

            if total_data_tokens > remaining_budget:
                data_items.sort(key=lambda pair: len(pair[1]), reverse=True)
                overflow = total_data_tokens - remaining_budget
                truncated: list[tuple[str, str]] = []
                for key, value in data_items:
                    if overflow <= 0:
                        truncated.append((key, value))
                        continue
                    current_tokens = _estimate_tokens(value)
                    reduction = min(current_tokens, overflow)
                    new_max = max(0, current_tokens - reduction)
                    truncated.append((key, _truncate_to_tokens(value, new_max)))
                    overflow -= reduction
                data_items = truncated

            for key, value in data_items:
                if value:
                    messages.append({
                        "role": "user",
                        "content": f"[Upstream data from {key}]:\n{value}",
                    })

        # 6. Tool results (never truncated)
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": str(result.get("content", "")),
            })

        # 7. User task (never truncated)
        messages.append({"role": "user", "content": user_task})

        return messages

    def validate_input(self, text: str) -> None:
        """Enforce all input-scope guardrail rules.

        Args:
            text: The raw user-facing input text to validate.

        Raises:
            GuardrailViolationError: If any input guardrail is violated.
        """
        violations = _check_rules(self._parsed_rules, "input", text)
        if violations:
            raise GuardrailViolationError(violations)

    def validate_output(self, text: str) -> None:
        """Enforce all output-scope guardrail rules.

        Args:
            text: The raw LLM output text to validate.

        Raises:
            GuardrailViolationError: If any output guardrail is violated.
        """
        violations = _check_rules(self._parsed_rules, "output", text)
        if violations:
            raise GuardrailViolationError(violations)



# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 characters).

    Args:
        text: Input string.

    Returns:
        Estimated token count.
    """
    return len(text) // _CHARS_PER_TOKEN


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens.

    Args:
        text: Input string.
        max_tokens: Upper bound in tokens.

    Returns:
        The (possibly shortened) string.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _parse_guardrail_rules(raw_rules: list[str]) -> list[GuardrailRule]:
    """Parse ``"scope:rule_type:value"`` strings into GuardrailRule objects.

    Args:
        raw_rules: List of raw guardrail strings.

    Returns:
        List of validated GuardrailRule instances.

    Raises:
        ValueError: On malformed scope, rule_type, or missing components.
    """
    parsed: list[GuardrailRule] = []
    for rule_str in raw_rules:
        parts = rule_str.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid guardrail format '{rule_str}': "
                "expected 'scope:rule_type:value'"
            )
        scope, rule_type, value = parts
        if scope not in ("input", "output"):
            raise ValueError(
                f"Invalid guardrail scope '{scope}' in '{rule_str}': "
                "must be 'input' or 'output'"
            )
        valid_types = (
            _VALID_INPUT_TYPES if scope == "input" else _VALID_OUTPUT_TYPES
        )
        if rule_type not in valid_types:
            raise ValueError(
                f"Invalid rule_type '{rule_type}' for scope '{scope}' "
                f"in '{rule_str}': must be one of {sorted(valid_types)}"
            )
        parsed.append(GuardrailRule(scope=scope, rule_type=rule_type, value=value))
    return parsed


def _check_rules(
    rules: list[GuardrailRule],
    scope: str,
    text: str,
) -> list[str]:
    """Evaluate all guardrail rules for a given scope against *text*.

    Args:
        rules: Full list of parsed guardrail rules.
        scope: ``"input"`` or ``"output"``.
        text: The text to validate.

    Returns:
        A list of human-readable violation descriptions (empty if clean).
    """
    violations: list[str] = []
    for rule in rules:
        if rule.scope != scope:
            continue

        if rule.rule_type == "max_length":
            limit = int(rule.value)
            if len(text) > limit:
                violations.append(
                    f"{scope.capitalize()} exceeds max_length {limit} "
                    f"(actual: {len(text)})"
                )

        elif rule.rule_type == "forbidden_topic":
            if rule.value.lower() in text.lower():
                violations.append(
                    f"{scope.capitalize()} contains forbidden topic: "
                    f"'{rule.value}'"
                )

        elif rule.rule_type == "forbidden_phrase":
            if rule.value.lower() in text.lower():
                violations.append(
                    f"{scope.capitalize()} contains forbidden phrase: "
                    f"'{rule.value}'"
                )

        elif rule.rule_type == "required_pattern":
            if not re.search(rule.value, text):
                violations.append(
                    f"{scope.capitalize()} does not match required pattern: "
                    f"'{rule.value}'"
                )

        elif rule.rule_type == "required_structure":
            if rule.value == "json":
                try:
                    json.loads(text)
                except json.JSONDecodeError as exc:
                    violations.append(
                        f"{scope.capitalize()} is not valid JSON: {exc}"
                    )

    return violations
