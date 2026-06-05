"""
API router for user settings and API-key management.

What it does:
    Exposes GET / PUT endpoints to read and replace general user settings
    (LLM providers, API credentials, defaults) and dedicated endpoints for
    managing LLM API keys.  API keys are persisted to the project-root
    ``.env`` file and set in ``os.environ`` for immediate use — they are
    never stored inside ``data/settings.json``.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/settings``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(persistence)`` during lifespan startup and
    includes the returned router in the FastAPI app.
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

from backend.settings.models import UserSettings, resolve_provider_api_key
from backend.settings.persistence import SettingsPersistence, persist_env_var_to_dotenv

router = APIRouter(prefix="/api/settings", tags=["settings"])

_persistence: SettingsPersistence | None = None


def init(persistence: SettingsPersistence) -> APIRouter:
    """Wire backend dependencies and return the configured router.

    Args:
        persistence: SettingsPersistence for JSON read/write.

    Returns:
        The fully configured APIRouter.
    """
    global _persistence  # noqa: PLW0603
    _persistence = persistence
    return router


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ApiKeySetRequest(BaseModel):
    """Payload for the set-API-key endpoint.

    Attributes:
        provider_name: Which provider the key belongs to (must match a
            configured ``LLMProviderConfig.provider_name``).
        api_key: The secret key value.
    """

    provider_name: str
    api_key: str


class ProviderKeyStatus(BaseModel):
    """Per-provider status returned by the provider-status endpoint.

    Attributes:
        provider_name: Provider identifier.
        api_key_env: Name of the environment variable.
        configured: ``True`` when the env var holds a non-empty value.
        masked_key: First-4 / last-4 masked preview, or empty if not set.
    """

    provider_name: str
    api_key_env: str
    configured: bool
    masked_key: str


# ---------------------------------------------------------------------------
# Endpoints — general settings (no secrets)
# ---------------------------------------------------------------------------

@router.get("/")
@router.get("")
async def get_settings() -> UserSettings:
    """Return the current user settings.

    Returns:
        The stored UserSettings object.
    """
    return _persistence.load_settings()


@router.put("/")
@router.put("")
async def update_settings(settings: UserSettings) -> UserSettings:
    """Replace the user settings wholesale.

    Args:
        settings: The complete UserSettings object to persist.

    Returns:
        The persisted UserSettings.
    """
    _persistence.save_settings(settings)
    return settings


# ---------------------------------------------------------------------------
# Endpoints — API key management
# ---------------------------------------------------------------------------

@router.get("/provider-status")
async def provider_status() -> list[ProviderKeyStatus]:
    """Return the configured/missing status of every LLM provider's API key.

    Keys are resolved from ``os.environ`` — the actual secret is never
    returned.  A masked preview (first-4 + last-4 characters) is included
    when the key is set.

    Returns:
        A list of ``ProviderKeyStatus`` objects.
    """
    settings = _persistence.load_settings()
    result: list[ProviderKeyStatus] = []
    for provider in settings.llm_providers:
        key = resolve_provider_api_key(provider)
        masked = ""
        if len(key) > 8:
            masked = key[:4] + "••••" + key[-4:]
        result.append(ProviderKeyStatus(
            provider_name=provider.provider_name,
            api_key_env=provider.api_key_env,
            configured=bool(key),
            masked_key=masked,
        ))
    return result


@router.post("/api-key")
async def set_api_key(request: ApiKeySetRequest) -> list[ProviderKeyStatus]:
    """Store a cloud API key in ``os.environ`` and persist to ``.env``.

    The key is written to the project-root ``.env`` so workers and future
    server restarts pick it up.  It is also set in ``os.environ`` for
    immediate use by the running process.

    Args:
        request: Contains the provider name and the API key value.

    Returns:
        Updated provider-status list.
    """
    settings = _persistence.load_settings()
    target_provider = None
    for provider in settings.llm_providers:
        if provider.provider_name == request.provider_name:
            target_provider = provider
            break

    if target_provider is None:
        raise ValueError(
            f"Provider '{request.provider_name}' is not configured in settings. "
            f"Available providers: "
            f"{[p.provider_name for p in settings.llm_providers]}"
        )

    env_var_name = target_provider.api_key_env
    os.environ[env_var_name] = request.api_key
    persist_env_var_to_dotenv(env_var_name, request.api_key)

    return await provider_status()
