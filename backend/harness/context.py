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
        system_prompt: Fixed system-level instruction for the LLM.
        agent_rules: Additional behavioural rules appended to the system message.
        token_budget: Maximum number of tokens for the assembled message list.
        scope_window: Maximum number of few-shot examples to retain.
        state: Mutable key/value store carried across invocations.
        few_shot_examples: Example user/assistant pairs for in-context learning.

    Methods:
        assemble_messages: Build the full LLM message list.
        validate_input: Enforce input guardrail rules (raises on violation).
        validate_output: Enforce output guardrail rules (raises on violation).
        update_state: Set a single key in the internal state dict.
        merge_upstream_state: Merge another dict into the internal state.
        get_state: Return a shallow copy of the current state.
    """

    def __init__(
        self,
        system_prompt: str,
        agent_rules: list[str],
        token_budget: int,
        scope_window: int,
        guardrail_rules: list[str],
        state: dict[str, Any] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> None:
        """Initialise the context harness.

        Args:
            system_prompt: System-level instruction for the LLM.
            agent_rules: Behavioural rules appended to the system message.
            token_budget: Token ceiling (1 token ≈ 4 chars).
            scope_window: How many recent few-shot examples to keep.
            guardrail_rules: Raw strings in ``"scope:rule_type:value"`` format.
            state: Optional initial state dict.
            few_shot_examples: Optional list of ``{"user": …, "assistant": …}``
                dicts.

        Raises:
            ValueError: If any guardrail string is malformed.
        """
        self.system_prompt = system_prompt
        self.agent_rules = list(agent_rules)
        self.token_budget = token_budget
        self.scope_window = scope_window
        self.state: dict[str, Any] = dict(state) if state else {}
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

        # 1. System message (never truncated)
        system_content = self.system_prompt
        if self.agent_rules:
            rules_block = "\n".join(f"- {rule}" for rule in self.agent_rules)
            system_content += f"\n\nRules:\n{rules_block}"
        messages.append({"role": "system", "content": system_content})

        # 2. Few-shot examples (windowed, never truncated)
        windowed_examples = self.few_shot_examples[-self.scope_window :]
        for example in windowed_examples:
            if "user" in example:
                messages.append({"role": "user", "content": example["user"]})
            if "assistant" in example:
                messages.append({"role": "assistant", "content": example["assistant"]})

        # 3. Compute fixed-cost tokens (system + examples + tool results + task)
        fixed_tokens = _estimate_tokens(system_content) + _estimate_tokens(user_task)
        for ex in windowed_examples:
            fixed_tokens += _estimate_tokens(ex.get("user", ""))
            fixed_tokens += _estimate_tokens(ex.get("assistant", ""))
        for tr in tool_results:
            fixed_tokens += _estimate_tokens(str(tr.get("content", "")))

        remaining_budget = max(0, self.token_budget - fixed_tokens)

        # 4. Truncate upstream data (largest first) to fit remaining budget
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

        # 5. Tool results (never truncated)
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": str(result.get("content", "")),
            })

        # 6. User task (never truncated)
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

    def update_state(self, key: str, value: Any) -> None:
        """Set a single key in the internal state dict.

        Args:
            key: State key.
            value: Arbitrary value to store.
        """
        self.state[key] = value

    def merge_upstream_state(self, upstream_state: dict[str, Any]) -> None:
        """Merge upstream state into the internal state dict.

        Args:
            upstream_state: Key/value pairs to merge (overwrites on collision).
        """
        self.state.update(upstream_state)

    def get_state(self) -> dict[str, Any]:
        """Return a shallow copy of the current state.

        Returns:
            A dict snapshot of the harness state.
        """
        return dict(self.state)


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
