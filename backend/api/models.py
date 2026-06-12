"""
API router for LLM model and tool discovery.

What it does:
    Lists available LLM models grouped by provider (with per-provider
    errors included in the response rather than silently skipped),
    lists models for a single provider (for the node config panel),
    and lists all registered tools with their parameter schemas.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/models``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(settings_persistence, tool_registry)``
    during lifespan startup and includes the returned router in the app.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.agent.localhost_resolver import resolve_localhost_url
from backend.agent.model_discovery import fetch_models_for_provider
from backend.settings.models import resolve_provider_api_key, resolve_provider_base_url
from backend.settings.persistence import SettingsPersistence
from backend.tools.registry import ToolRegistry

router = APIRouter(prefix="/api/models", tags=["models"])

_settings_persistence: SettingsPersistence | None = None
_tool_registry: ToolRegistry | None = None


def init(
    settings_persistence: SettingsPersistence,
    tool_registry: ToolRegistry,
) -> APIRouter:
    """Wire backend dependencies and return the configured router."""
    global _settings_persistence, _tool_registry  # noqa: PLW0603
    _settings_persistence = settings_persistence
    _tool_registry = tool_registry
    return router


@router.get("/")
@router.get("")
async def list_models() -> dict[str, Any]:
    """List LLM models available from each configured provider."""
    settings = _settings_persistence.load_settings()
    providers: list[dict[str, Any]] = []

    for config in settings.llm_providers:
        provider_info: dict[str, Any] = {
            "provider_name": config.provider_name,
        }

        api_key = resolve_provider_api_key(config)
        base_url = resolve_provider_base_url(config)

        if not base_url:
            provider_info["models"] = []
            provider_info["error"] = (
                f"No base URL configured for provider '{config.provider_name}'. "
                f"Set it in the Settings page."
            )
            providers.append(provider_info)
            continue

        if not api_key and not config.is_local:
            provider_info["models"] = []
            provider_info["error"] = (
                f"API key not configured for provider '{config.provider_name}'. "
                f"Set it via the Settings > API Keys page."
            )
            providers.append(provider_info)
            continue

        if config.is_local:
            base_url = resolve_localhost_url(base_url)

        try:
            provider = LLMProvider(
                api_key=api_key or "dummy",
                model_id="",
                provider=config.provider_name,
                base_url=base_url,
            )
            models = await provider.list_models()
            provider_info["models"] = models
            provider_info["error"] = None
        except (LLMProviderError, Exception) as exc:
            provider_info["models"] = []
            provider_info["error"] = str(exc)
        providers.append(provider_info)

    return {"providers": providers}


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    """List all registered tools with their parameter schemas and hierarchy."""
    return {
        "tools": _tool_registry.get_tool_definitions(),
        "hierarchy": _tool_registry.get_tool_hierarchy(),
    }


@router.get("/{provider_name}")
async def list_models_for_provider(provider_name: str) -> dict[str, Any]:
    """List LLM models available from a single provider.

    Returns {provider_name, models: [{id, label}, ...], error}.
    Resolves provider from the fixed known sets (CLOUD_PROVIDER_API_BASE,
    LOCAL_ENDPOINT_ENV_VAR) — does NOT require provider to be in settings.json.
    """
    from backend.settings.models import (
        CLOUD_PROVIDER_API_BASE,
        LOCAL_ENDPOINT_ENV_VAR,
        LOCAL_PROVIDERS,
        PROVIDER_DEFAULT_ENV_VAR,
    )

    is_local = provider_name in LOCAL_PROVIDERS
    is_cloud = provider_name in CLOUD_PROVIDER_API_BASE

    if not is_local and not is_cloud:
        return {
            "provider_name": provider_name,
            "models": [],
            "error": f"Unknown provider '{provider_name}'.",
        }

    if is_cloud:
        base_url = CLOUD_PROVIDER_API_BASE[provider_name]
        env_var = PROVIDER_DEFAULT_ENV_VAR.get(provider_name, "")
        api_key = os.environ.get(env_var, "").strip() if env_var else ""
        if not api_key:
            return {
                "provider_name": provider_name,
                "models": [],
                "error": f"API key not configured. Set {env_var} in Settings.",
            }
    else:
        env_var = LOCAL_ENDPOINT_ENV_VAR[provider_name]
        base_url = os.environ.get(env_var, "").strip()
        if not base_url:
            return {
                "provider_name": provider_name,
                "models": [],
                "error": f"Endpoint not configured. Set URL in Settings.",
            }
        base_url = resolve_localhost_url(base_url)
        api_key = "dummy"

    try:
        models = await fetch_models_for_provider(
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
        )
        return {
            "provider_name": provider_name,
            "models": models,
            "error": None,
        }
    except Exception as exc:
        return {
            "provider_name": provider_name,
            "models": [],
            "error": str(exc),
        }
