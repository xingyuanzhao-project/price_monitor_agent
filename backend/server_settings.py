"""Centralised env-driven configuration for the server process.

Every tunable read by ``backend.main`` lives on :class:`ServerSettings`.
Settings are populated from environment variables prefixed with
``PRICE_MONITOR_`` so deployment configuration stays out of source code.

Ported from nocode-workflow/server/settings.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class ServerSettings(BaseSettings):
    """Process-wide configuration for the server.

    Attributes:
        host: Bind address for uvicorn.
        port: Bind port for uvicorn.
        cors_origins: Origins allowed by CORSMiddleware.
        log_level: Root logger level.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRICE_MONITOR_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8000"]
    )
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_server_settings() -> ServerSettings:
    """Return the process-wide cached ServerSettings instance."""
    return ServerSettings()
