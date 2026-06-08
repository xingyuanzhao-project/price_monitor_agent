"""
Settings subpackage for user configuration and credential management.

What it does:
    Defines data models for API credentials, LLM provider configurations, and
    user settings. Provides JSON-based persistence for loading and saving these
    configurations to disk.

Entities in it:
    - models: Pydantic models for APICredential, LLMProviderConfig, and UserSettings.
    - persistence: SettingsPersistence class for JSON-based save/load operations.

How used by other modules:
    - The tools subpackage retrieves API credentials via UserSettings to authenticate
      external service calls.
    - The agent subpackage retrieves LLM provider configurations to instantiate
      LLMProvider objects with correct API keys and base URLs.
    - The frontend calls persistence methods to save user-entered configuration.
"""

from backend.settings.models import (
    CLOUD_PROVIDER_API_BASE,
    LOCAL_ENDPOINT_ENV_VAR,
    LOCAL_PROVIDERS,
    PROVIDER_AUTH_TEST_URLS,
    PROVIDER_DEFAULT_ENV_VAR,
    APICredential,
    LLMProviderConfig,
    UserSettings,
    resolve_provider_api_key,
    resolve_provider_base_url,
)
from backend.settings.persistence import SettingsPersistence
