"""
FastAPI application entry point with lifespan-managed startup.

What it does:
    Configures the complete backend application: loads environment variables,
    initialises persistence layers, registers all 15 tools, creates the
    workflow executor, wires API routers, enables CORS, and optionally
    mounts a static frontend build.

Entities in it:
    - lifespan: Async context manager performing startup / shutdown.
    - app: The FastAPI application instance.

How used by other modules:
    Run directly via ``uvicorn backend.main:app`` or imported for testing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import models as models_router
from backend.api import runs as runs_router
from backend.api import schemas as schemas_router
from backend.api import settings as settings_router
from backend.orchestration.executor import WorkflowExecutor
from backend.schema.persistence import SchemaPersistence
from backend.schema.validation import SchemaValidator
from backend.settings.models import PROVIDER_DEFAULT_ENV_VAR, LLMProviderConfig, UserSettings
from backend.settings.persistence import SettingsPersistence
from backend.tools.alert_dispatch import SendEmailTool, SendTelegramTool, SendWebhookTool
from backend.tools.data_acquisition import FetchDataTool
from backend.tools.write_output import WriteOutputTool
from backend.tools.registry import ToolRegistry
from backend.tools.technical_analysis import (
    ComputeIndicatorTool,
    ComputeStatisticTool,
    DetectSignalTool,
)
from backend.tools.text_analysis import (
    ChunkTextTool,
    ClassifyTextTool,
    CrossModalAlignmentTool,
    ExtractEntitiesTool,
    ScoreTextTool,
    SemanticSearchTool,
    SummarizeTextTool,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise all backend components on startup.

    Performs the following in order:
        1. Load ``.env`` so API keys are available in ``os.environ``.
        2. Create the ``schemas/`` directory and a SchemaPersistence.
        3. Create a SettingsPersistence; if no settings file exists yet,
           seed default settings with an OpenRouter provider entry
           (referencing the ``OPENROUTER_API_KEY`` env var — the actual
           secret is never written to settings JSON).
        4. Register all 15 tool instances in a ToolRegistry.
        5. Create a WorkflowExecutor.
        6. Wire every API router with its dependencies.

    Args:
        app: The FastAPI application being started.

    Yields:
        Control back to the ASGI server for the application's lifetime.
    """
    load_dotenv()

    # -- schema persistence -------------------------------------------------
    schemas_dir = Path("schemas")
    schemas_dir.mkdir(exist_ok=True)
    schema_persistence = SchemaPersistence(schemas_dir)
    validator = SchemaValidator()

    # -- settings persistence -----------------------------------------------
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    settings_persistence = SettingsPersistence(Path("data/settings.json"))

    try:
        user_settings = settings_persistence.load_settings()
    except FileNotFoundError:
        user_settings = UserSettings(
            llm_providers=[
                LLMProviderConfig(
                    provider_name="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                    api_key_env=PROVIDER_DEFAULT_ENV_VAR["openrouter"],
                    available_models=[],
                ),
            ],
        )
        settings_persistence.save_settings(user_settings)

    # -- tool registry (all 15 tools) ---------------------------------------
    tool_registry = ToolRegistry()
    for tool_instance in [
        FetchDataTool(),
        ComputeIndicatorTool(),
        ComputeStatisticTool(),
        DetectSignalTool(),
        ChunkTextTool(),
        SemanticSearchTool(),
        ExtractEntitiesTool(),
        ClassifyTextTool(),
        ScoreTextTool(),
        SummarizeTextTool(),
        CrossModalAlignmentTool(),
        SendWebhookTool(),
        SendEmailTool(),
        SendTelegramTool(),
        WriteOutputTool(),
    ]:
        tool_registry.register(tool_instance)

    # -- workflow executor --------------------------------------------------
    executor = WorkflowExecutor(tool_registry, user_settings)

    # -- wire API routers ---------------------------------------------------
    schemas_router.init(schema_persistence, validator)
    runs_router.init(executor, schema_persistence)
    settings_router.init(settings_persistence)
    models_router.init(settings_persistence, tool_registry)

    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Price Monitor Agent",
    description=(
        "Multi-agent workflow system for financial market monitoring "
        "and analysis."
    ),
    version="0.1.0",
    lifespan=lifespan,
    redirect_slashes=True,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(schemas_router.router)
app.include_router(runs_router.router)
app.include_router(settings_router.router)
app.include_router(models_router.router)

frontend_dir = Path(__file__).parent / "static"
if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )
