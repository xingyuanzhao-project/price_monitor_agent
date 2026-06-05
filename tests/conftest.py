"""Shared fixtures for end-to-end tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from backend.tools.registry import ToolRegistry
from backend.tools.data_acquisition import FetchDataTool
from backend.tools.technical_analysis import (
    ComputeIndicatorTool,
    ComputeStatisticTool,
    DetectSignalTool,
)
from backend.tools.text_analysis import (
    ChunkTextTool,
    SemanticSearchTool,
    ExtractEntitiesTool,
    ClassifyTextTool,
    ScoreTextTool,
    SummarizeTextTool,
    CrossModalAlignmentTool,
)
from backend.tools.alert_dispatch import SendWebhookTool, SendEmailTool, SendTelegramTool
from backend.tools.write_output import WriteOutputTool
from backend.settings.models import UserSettings, LLMProviderConfig
from backend.orchestration.executor import WorkflowExecutor
from backend.schema.persistence import SchemaPersistence
from backend.schema.validation import SchemaValidator

TEST_MODEL_ID = "google/gemini-2.5-flash-lite"


@pytest.fixture
def openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture
def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
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
        registry.register(tool_instance)
    return registry


@pytest.fixture
def user_settings(openrouter_api_key: str) -> UserSettings:
    return UserSettings(
        llm_providers=[
            LLMProviderConfig(
                provider_name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                available_models=[TEST_MODEL_ID],
            ),
        ],
        global_defaults={
            "max_tokens": 32768,
            "rate_limit_per_minute": 30,
        },
    )


@pytest.fixture
def workflow_executor(
    tool_registry: ToolRegistry, user_settings: UserSettings
) -> WorkflowExecutor:
    return WorkflowExecutor(tool_registry, user_settings)


@pytest.fixture
def schema_persistence(tmp_path: Path) -> SchemaPersistence:
    return SchemaPersistence(tmp_path)


@pytest.fixture
def schema_validator() -> SchemaValidator:
    return SchemaValidator()
