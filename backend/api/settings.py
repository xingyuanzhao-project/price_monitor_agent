"""
API router for user settings, API-key management, and local endpoint management.

What it does:
    Exposes GET / PUT endpoints to read and replace general user settings,
    dedicated endpoints for managing LLM API keys (cloud providers), and
    endpoints for managing local model server endpoints (Ollama, vLLM, llama.cpp).
    Includes test endpoints that validate keys/endpoints before persisting.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/settings``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(persistence)`` during lifespan startup and
    includes the returned router in the FastAPI app.
"""

from __future__ import annotations

import os
from typing import List

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from backend.agent.localhost_resolver import resolve_localhost_url
from backend.settings.models import (
    LOCAL_ENDPOINT_ENV_VAR,
    LOCAL_PROVIDERS,
    PROVIDER_AUTH_TEST_URLS,
    PROVIDER_DEFAULT_ENV_VAR,
    UserSettings,
    resolve_provider_api_key,
)
from backend.settings.persistence import SettingsPersistence, persist_env_var_to_dotenv

router = APIRouter(prefix="/api/settings", tags=["settings"])

_persistence: SettingsPersistence | None = None


def init(persistence: SettingsPersistence) -> APIRouter:
    """Wire backend dependencies and return the configured router."""
    global _persistence  # noqa: PLW0603
    _persistence = persistence
    return router


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ApiKeySetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_name: str
    api_key: str


class ApiKeyTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_name: str
    api_key: str


class ApiKeyTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_name: str
    valid: bool
    message: str


class LocalEndpointSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_name: str
    api_base: str


class LocalEndpointTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_base: str


class LocalEndpointTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reachable: bool
    models: List[str] = []
    message: str


class ProviderKeyStatus(BaseModel):
    provider_name: str
    api_key_env: str
    configured: bool
    masked_key: str


class LocalEndpointStatus(BaseModel):
    provider_name: str
    api_base: str
    configured: bool


class ProviderStatusResponse(BaseModel):
    cloud_providers: List[ProviderKeyStatus]
    local_endpoints: List[LocalEndpointStatus]


# ---------------------------------------------------------------------------
# Endpoints — general settings
# ---------------------------------------------------------------------------

@router.get("/")
@router.get("")
async def get_settings() -> UserSettings:
    """Return the current user settings."""
    return _persistence.load_settings()


@router.put("/")
@router.put("")
async def update_settings(settings: UserSettings) -> UserSettings:
    """Replace the user settings wholesale."""
    _persistence.save_settings(settings)
    return settings


# ---------------------------------------------------------------------------
# Endpoints — provider status (cloud + local)
# ---------------------------------------------------------------------------

@router.get("/provider-status")
async def provider_status() -> ProviderStatusResponse:
    """Return cloud key status and local endpoint status.

    Cloud providers are the fixed known set (from PROVIDER_DEFAULT_ENV_VAR),
    not from user config. This matches nocode-workflow: all providers always
    show, user just sets the key.
    """
    cloud_items: list[ProviderKeyStatus] = []
    for provider_name, env_var in PROVIDER_DEFAULT_ENV_VAR.items():
        key = os.environ.get(env_var, "").strip()
        masked = ""
        if len(key) > 8:
            masked = key[:4] + "••••" + key[-4:]
        cloud_items.append(ProviderKeyStatus(
            provider_name=provider_name,
            api_key_env=env_var,
            configured=bool(key),
            masked_key=masked,
        ))

    local_items: list[LocalEndpointStatus] = []
    for provider_name, env_var in LOCAL_ENDPOINT_ENV_VAR.items():
        configured_url = os.environ.get(env_var, "").strip()
        local_items.append(LocalEndpointStatus(
            provider_name=provider_name,
            api_base=configured_url,
            configured=bool(configured_url),
        ))

    return ProviderStatusResponse(
        cloud_providers=cloud_items,
        local_endpoints=local_items,
    )


# ---------------------------------------------------------------------------
# Endpoints — cloud API key management
# ---------------------------------------------------------------------------

@router.post("/api-key")
async def set_api_key(request: ApiKeySetRequest) -> ProviderStatusResponse:
    """Store a cloud API key in ``os.environ`` and persist to ``.env``."""
    env_var_name = PROVIDER_DEFAULT_ENV_VAR.get(request.provider_name)
    if not env_var_name:
        settings = _persistence.load_settings()
        target = None
        for p in settings.llm_providers:
            if p.provider_name == request.provider_name:
                target = p
                break
        if target is None or not target.api_key_env:
            raise ValueError(
                f"Provider '{request.provider_name}' is not configured. "
                f"Known cloud providers: {list(PROVIDER_DEFAULT_ENV_VAR.keys())}"
            )
        env_var_name = target.api_key_env

    os.environ[env_var_name] = request.api_key
    persist_env_var_to_dotenv(env_var_name, request.api_key)
    return await provider_status()


@router.post("/api-key/test")
async def test_api_key(request: ApiKeyTestRequest) -> ApiKeyTestResponse:
    """Validate a cloud API key against the provider's auth endpoint."""
    if request.provider_name not in PROVIDER_AUTH_TEST_URLS:
        return ApiKeyTestResponse(
            provider_name=request.provider_name,
            valid=False,
            message=f"No test endpoint known for provider '{request.provider_name}'.",
        )

    test_url, headers = _build_test_request(request.provider_name, request.api_key)
    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            response = await http_client.get(test_url, headers=headers)
        if response.status_code == 200:
            return ApiKeyTestResponse(
                provider_name=request.provider_name,
                valid=True,
                message=f"{request.provider_name} key is valid.",
            )
        if response.status_code in (401, 403):
            return ApiKeyTestResponse(
                provider_name=request.provider_name,
                valid=False,
                message=f"{request.provider_name} rejected the key (HTTP {response.status_code}).",
            )
        return ApiKeyTestResponse(
            provider_name=request.provider_name,
            valid=False,
            message=(
                f"{request.provider_name} returned HTTP {response.status_code}. "
                "The key may be invalid or the service may be unavailable."
            ),
        )
    except httpx.RequestError as connection_error:
        return ApiKeyTestResponse(
            provider_name=request.provider_name,
            valid=False,
            message=f"Connection failed: {connection_error}",
        )


def _build_test_request(provider: str, api_key: str) -> tuple[str, dict[str, str]]:
    """Return (url, headers) for the provider's key-validation request."""
    base_url = PROVIDER_AUTH_TEST_URLS[provider]
    if provider == "anthropic":
        return base_url, {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    if provider == "google":
        return f"{base_url}?key={api_key}", {}
    return base_url, {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Endpoints — local endpoint management
# ---------------------------------------------------------------------------

@router.post("/local-endpoint")
async def set_local_endpoint(request: LocalEndpointSetRequest) -> ProviderStatusResponse:
    """Resolve and persist a local endpoint URL."""
    if request.provider_name not in LOCAL_ENDPOINT_ENV_VAR:
        raise ValueError(
            f"'{request.provider_name}' is not a local provider. "
            f"Known local providers: {list(LOCAL_ENDPOINT_ENV_VAR.keys())}"
        )

    env_var = LOCAL_ENDPOINT_ENV_VAR[request.provider_name]
    resolved_url = resolve_localhost_url(request.api_base.rstrip("/"))
    os.environ[env_var] = resolved_url
    persist_env_var_to_dotenv(env_var, resolved_url)
    return await provider_status()


@router.post("/local-endpoint/test")
async def test_local_endpoint(request: LocalEndpointTestRequest) -> LocalEndpointTestResponse:
    """Test a local endpoint by hitting its ``/models`` path."""
    base = resolve_localhost_url(request.api_base.rstrip("/"))
    models_url = f"{base}/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(
                models_url,
                headers={"Authorization": "Bearer dummy"},
            )
        if response.status_code == 200:
            body = response.json()
            model_ids: list[str] = []
            if isinstance(body, dict) and "data" in body:
                for entry in body["data"]:
                    if isinstance(entry, dict) and "id" in entry:
                        model_ids.append(str(entry["id"]))
            return LocalEndpointTestResponse(
                reachable=True,
                models=model_ids,
                message=(
                    f"Connected. Found {len(model_ids)} model(s): "
                    f"{', '.join(model_ids[:5]) or 'none listed'}."
                ),
            )
        return LocalEndpointTestResponse(
            reachable=False,
            models=[],
            message=f"Endpoint returned HTTP {response.status_code}.",
        )
    except httpx.RequestError as connection_error:
        return LocalEndpointTestResponse(
            reachable=False,
            models=[],
            message=f"Connection failed: {connection_error}",
        )


# ---------------------------------------------------------------------------
# Data Sources endpoints
# ---------------------------------------------------------------------------

@router.get("/data-sources")
async def get_data_sources() -> dict:
    """Return public and additional data source registries plus user config."""
    from backend.tools.supports.registry import (
        get_public_sources_by_category,
        get_additional_sources_by_category,
    )
    settings = _persistence.load_settings()
    return {
        "public_sources": get_public_sources_by_category(),
        "additional_sources": get_additional_sources_by_category(),
        "enabled_public": settings.enabled_public_sources,
        "configured_additional": [
            {"source_id": api.source_id, "api_key": "***" if api.api_key else "", "base_url": api.base_url}
            for api in settings.additional_tool_apis
        ],
    }


class TogglePublicSourceBody(BaseModel):
    model_config = ConfigDict(strict=True)
    source_id: str
    enabled: bool


@router.post("/data-sources/public/toggle")
async def toggle_public_source(body: TogglePublicSourceBody) -> dict:
    """Enable or disable a public data source."""
    settings = _persistence.load_settings()
    if body.enabled and body.source_id not in settings.enabled_public_sources:
        settings.enabled_public_sources.append(body.source_id)
    elif not body.enabled and body.source_id in settings.enabled_public_sources:
        settings.enabled_public_sources.remove(body.source_id)
    _persistence.save_settings(settings)
    return {"status": "ok", "enabled_public": settings.enabled_public_sources}


class BatchTogglePublicSourceBody(BaseModel):
    model_config = ConfigDict(strict=True)
    source_ids: list[str]
    enabled: bool


@router.post("/data-sources/public/toggle-batch")
async def toggle_public_sources_batch(body: BatchTogglePublicSourceBody) -> dict:
    """Enable or disable multiple public data sources at once."""
    settings = _persistence.load_settings()
    current = set(settings.enabled_public_sources)
    if body.enabled:
        current.update(body.source_ids)
    else:
        current -= set(body.source_ids)
    settings.enabled_public_sources = sorted(current)
    _persistence.save_settings(settings)
    return {"status": "ok", "enabled_public": settings.enabled_public_sources}


class AddAdditionalApiBody(BaseModel):
    model_config = ConfigDict(strict=True)
    source_id: str
    api_key: str = ""
    base_url: str = ""


@router.post("/data-sources/additional")
async def add_additional_api(body: AddAdditionalApiBody) -> dict:
    """Add or update an additional tool API configuration."""
    from backend.settings.models import AdditionalToolApi
    settings = _persistence.load_settings()
    existing = next((a for a in settings.additional_tool_apis if a.source_id == body.source_id), None)
    if existing:
        existing.api_key = body.api_key
        existing.base_url = body.base_url
    else:
        settings.additional_tool_apis.append(
            AdditionalToolApi(source_id=body.source_id, api_key=body.api_key, base_url=body.base_url)
        )
    _persistence.save_settings(settings)
    return {"status": "ok"}


@router.delete("/data-sources/additional/{source_id}")
async def remove_additional_api(source_id: str) -> dict:
    """Remove an additional tool API configuration."""
    settings = _persistence.load_settings()
    settings.additional_tool_apis = [a for a in settings.additional_tool_apis if a.source_id != source_id]
    _persistence.save_settings(settings)
    return {"status": "ok"}
