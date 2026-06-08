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
from backend.run_log_handler import install_handler as install_run_log_handler
from backend.orchestration.executor import WorkflowExecutor
from backend.schema.persistence import SchemaPersistence
from backend.schema.validation import SchemaValidator
from backend.server_settings import get_server_settings
from backend.settings.models import PROVIDER_DEFAULT_ENV_VAR, LLMProviderConfig, UserSettings
from backend.settings.persistence import SettingsPersistence
from backend.tools.alert_dispatch import SendEmailTool, SendTelegramTool, SendWebhookTool
from backend.tools.data_acquisition import FetchDataTool
from backend.tools.write_output import WriteOutputTool
from backend.tools.registry import ToolRegistry
from backend.tools.backtest import (
    DetectRegimeTool,
    EstimateParametersTool,
    RunMonteCarloTool,
    SimulateProcessTool,
)
from backend.tools.financial_analysis import (
    TechnicalAnalysisTool,
    QuantitativeAnalysisTool,
    SignalAnalysisTool,
    DiagnosticAnalysisTool,
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
    """Application lifespan: initialise all backend components on startup."""
    load_dotenv()
    install_run_log_handler()

    schemas_dir = Path("schemas")
    schemas_dir.mkdir(exist_ok=True)
    schema_persistence = SchemaPersistence(schemas_dir)
    validator = SchemaValidator()

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
                    base_url="",
                    api_key_env=PROVIDER_DEFAULT_ENV_VAR["openrouter"],
                    available_models=[],
                ),
            ],
        )
        settings_persistence.save_settings(user_settings)

    tool_registry = ToolRegistry()
    for tool_instance in [
        FetchDataTool(),
        TechnicalAnalysisTool(),
        QuantitativeAnalysisTool(),
        SignalAnalysisTool(),
        DiagnosticAnalysisTool(),
        DetectRegimeTool(),
        EstimateParametersTool(),
        SimulateProcessTool(),
        RunMonteCarloTool(),
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

    executor = WorkflowExecutor(tool_registry, user_settings)

    schemas_router.init(schema_persistence, validator)
    runs_router.init(executor, schema_persistence)
    settings_router.init(settings_persistence)
    models_router.init(settings_persistence, tool_registry)

    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

server_settings = get_server_settings()

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
    allow_origins=server_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(schemas_router.router)
app.include_router(runs_router.router)
app.include_router(settings_router.router)
app.include_router(models_router.router)


@app.get("/api/health", tags=["health"])
async def health_check() -> dict:
    """Return a simple health status for the server."""
    return {"status": "ok"}


frontend_dir = Path(__file__).parent / "static"
if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )
