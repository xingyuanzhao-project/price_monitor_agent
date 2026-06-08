"""Request normalization for provider-specific LLM API differences.

- :func:`normalize_request_kwargs` — normalizes the ``response_format``
  parameter shape for providers whose OpenAI-compatible endpoints accept
  structured output but with a different parameter form.

  Google/Gemini's endpoint supports ``{"type": "json_object"}`` but does
  not reliably handle ``{"type": "json_schema", "json_schema": {...}}``;
  the normalizer converts it to json_object mode + schema in prompt.

  All other providers (openrouter, openai, anthropic, ollama, vllm,
  llama_cpp) accept the full json_schema form unchanged.

Ported from nocode-workflow/src/format_adapter.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def normalize_request_kwargs(
    provider: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize the response_format parameter shape for the target provider.

    This is format normalization, not capability gating — all providers
    receive response_format; only the parameter shape differs.
    """
    if provider != "google":
        return kwargs

    response_format = kwargs.get("response_format")
    if response_format is None:
        return kwargs

    if not isinstance(response_format, dict):
        return kwargs

    if response_format.get("type") != "json_schema":
        return kwargs

    adapted = dict(kwargs)
    adapted["response_format"] = {"type": "json_object"}

    schema_instruction = _build_schema_instruction(response_format)
    messages = list(adapted.get("messages", []))
    if messages and messages[0].get("role") == "system":
        original_system = messages[0]["content"]
        messages[0] = {
            "role": "system",
            "content": f"{original_system}\n\n{schema_instruction}",
        }
    else:
        messages.insert(0, {"role": "system", "content": schema_instruction})
    adapted["messages"] = messages
    return adapted


def _build_schema_instruction(response_format: Dict[str, Any]) -> str:
    """Convert a response_format json_schema dict into prompt instructions."""
    json_schema = response_format.get("json_schema", {})
    schema = json_schema.get("schema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = [
        "You MUST respond with ONLY a valid JSON object (no markdown, no explanation).",
        "The JSON object must have exactly these fields:",
    ]
    for field_name, field_spec in properties.items():
        field_type = field_spec.get("type", "string")
        req_marker = " (required)" if field_name in required else ""
        lines.append(f'  - "{field_name}": {field_type}{req_marker}')

    lines.append("")
    lines.append("Example format:")
    example = {}
    for field_name, field_spec in properties.items():
        ftype = field_spec.get("type", "string")
        if ftype == "array":
            example[field_name] = []
        elif ftype == "object":
            example[field_name] = {}
        else:
            example[field_name] = "..."
    lines.append(json.dumps(example, indent=2))

    return "\n".join(lines)
