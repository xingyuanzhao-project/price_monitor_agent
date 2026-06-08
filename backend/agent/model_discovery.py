"""Per-provider model catalogue fetching and normalisation.

Fetches the upstream model list for each provider and normalises the
response into ``{id, label}`` entries. Each provider returns a different
JSON shape; this module handles the differences.

Ported from nocode-workflow/server/services/model_list_proxy.py.
"""

from __future__ import annotations

from typing import Any

import httpx


async def fetch_models_for_provider(
    provider_name: str,
    base_url: str,
    api_key: str,
) -> list[dict[str, str]]:
    """Fetch and normalise the model list for a provider.

    Args:
        provider_name: e.g. "openrouter", "openai", "ollama"
        base_url: The resolved base URL for this provider.
        api_key: The API key (or "dummy" for local providers).

    Returns:
        List of {id, label} dicts sorted by id.
    """
    request_url, headers = _build_request(provider_name, base_url, api_key)

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(request_url, headers=headers)

    if response.status_code >= 400:
        raise RuntimeError(
            f"{provider_name} model list returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    body = response.json()
    entries = _normalise(provider_name, body)
    entries.sort(key=lambda e: e["id"])
    return entries


def _build_request(
    provider_name: str, base_url: str, api_key: str
) -> tuple[str, dict[str, str]]:
    """Return (url, headers) for the model list request."""
    if provider_name == "anthropic":
        return "https://api.anthropic.com/v1/models", {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    if provider_name == "google":
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            {},
        )
    return f"{base_url.rstrip('/')}/models", {
        "Authorization": f"Bearer {api_key}",
    }


def _normalise(provider_name: str, body: Any) -> list[dict[str, str]]:
    """Normalise raw upstream JSON into [{id, label}]."""
    if provider_name == "google":
        return _normalise_google(body)

    raw_entries = body.get("data") if isinstance(body, dict) else None
    if not isinstance(raw_entries, list):
        return []

    results: list[dict[str, str]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id:
            continue

        if provider_name == "openrouter":
            label = str(entry.get("name", model_id)).strip() or model_id
        elif provider_name == "anthropic":
            label = str(entry.get("display_name", model_id)).strip() or model_id
        else:
            label = model_id

        results.append({"id": model_id, "label": label})
    return results


def _normalise_google(body: Any) -> list[dict[str, str]]:
    """Google returns {models: [{name: "models/gemini-pro", displayName: ...}]}."""
    raw_entries = body.get("models") if isinstance(body, dict) else None
    if not isinstance(raw_entries, list):
        return []

    results: list[dict[str, str]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        raw_name = str(entry.get("name", "")).strip()
        if not raw_name:
            continue
        model_id = raw_name.removeprefix("models/")
        label = str(entry.get("displayName", model_id)).strip() or model_id
        results.append({"id": model_id, "label": label})
    return results
