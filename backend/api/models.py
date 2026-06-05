"""
API router for LLM model and tool discovery.

What it does:
    Lists available LLM models grouped by provider (with per-provider
    errors included in the response rather than silently skipped) and
    lists all registered tools with their parameter schemas.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/models``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(settings_persistence, tool_registry)``
    during lifespan startup and includes the returned router in the app.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.settings.models import resolve_provider_api_key
from backend.settings.persistence import SettingsPersistence
from backend.tools.registry import ToolRegistry

router = APIRouter(prefix="/api/models", tags=["models"])

_settings_persistence: SettingsPersistence | None = None
_tool_registry: ToolRegistry | None = None


def init(
    settings_persistence: SettingsPersistence,
    tool_registry: ToolRegistry,
) -> APIRouter:
    """Wire backend dependencies and return the configured router.

    Args:
        settings_persistence: For reading the current user settings.
        tool_registry: For listing registered tool instances.

    Returns:
        The fully configured APIRouter.
    """
    global _settings_persistence, _tool_registry  # noqa: PLW0603
    _settings_persistence = settings_persistence
    _tool_registry = tool_registry
    return router


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
@router.get("")
async def list_models() -> dict[str, Any]:
    """List LLM models available from each configured provider.

    Per-provider errors (network failures, auth issues, etc.) are captured
    and returned in the ``error`` field — they are **not** silently skipped.

    Returns:
        ``{"providers": [{"provider_name": …, "models": […], "error": …}, …]}``.
    """
    settings = _settings_persistence.load_settings()
    providers: list[dict[str, Any]] = []

    for config in settings.llm_providers:
        provider_info: dict[str, Any] = {
            "provider_name": config.provider_name,
        }
        api_key = resolve_provider_api_key(config)
        if not api_key:
            provider_info["models"] = []
            provider_info["error"] = (
                f"Environment variable '{config.api_key_env}' is not set. "
                f"Set it in .env or via the Settings > API Keys page."
            )
            providers.append(provider_info)
            continue
        try:
            provider = LLMProvider(
                api_key=api_key,
                model_id="",
                base_url=config.base_url,
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
    """List all registered tools with their parameter schemas.

    Returns:
        ``{"tools": [{"name": …, "description": …, "parameters_schema": …}, …]}``.
    """
    return {"tools": _tool_registry.get_tool_definitions()}
