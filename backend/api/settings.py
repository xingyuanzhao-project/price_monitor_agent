"""
API router for user settings management.

What it does:
    Exposes GET / PUT endpoints to read and replace the global user settings
    (LLM providers, API credentials, defaults).  Persistence is delegated
    to a SettingsPersistence instance injected via ``init``.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/settings``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(persistence)`` during lifespan startup and
    includes the returned router in the FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.settings.models import UserSettings
from backend.settings.persistence import SettingsPersistence

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
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def get_settings() -> UserSettings:
    """Return the current user settings.

    Returns:
        The stored UserSettings object.
    """
    return _persistence.load_settings()


@router.put("/")
async def update_settings(settings: UserSettings) -> UserSettings:
    """Replace the user settings wholesale.

    Args:
        settings: The complete UserSettings object to persist.

    Returns:
        The persisted UserSettings.
    """
    _persistence.save_settings(settings)
    return settings
