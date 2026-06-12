"""
Pydantic data models for user settings, credentials, and LLM provider configuration.

What it does:
    Defines structured types for API credentials (keyed by name and type),
    LLM provider connection details (endpoint, env-var reference, models),
    and a top-level UserSettings container that aggregates all configuration
    with lookup methods.  LLM API keys are never stored in config — they
    live in environment variables (typically loaded from ``.env``).

    Provider taxonomy splits providers into cloud (known base URLs) and local
    (user-configured base URLs for Ollama, vLLM, llama.cpp).  This follows
    the nocode-workflow pattern of first-class local model support.

Entities in it:
    - PROVIDER_DEFAULT_ENV_VAR: Mapping from cloud provider name to env-var name.
    - LOCAL_ENDPOINT_ENV_VAR: Mapping from local provider name to env-var name.
    - CLOUD_PROVIDER_API_BASE: Known base URLs for cloud providers.
    - LOCAL_PROVIDERS: Set of local provider identifiers.
    - PROVIDER_AUTH_TEST_URLS: Auth-gated endpoints for key validation.
    - APICredential: A named credential with a type classification and arbitrary fields.
    - LLMProviderConfig: Connection configuration for a single LLM provider.
    - UserSettings: Aggregated user settings with credentials, providers, and defaults.
    - resolve_provider_api_key: Resolve a provider's API key from the environment.
    - resolve_provider_base_url: Resolve a provider's base URL (cloud=known, local=env).

How used by other modules:
    - backend.tools.base injects credentials from UserSettings into tool instances.
    - backend.agent.llm_provider reads LLMProviderConfig to construct provider clients.
    - backend.settings.persistence serializes/deserializes UserSettings to/from JSON.
    - backend.api.settings uses resolve_provider_api_key for runtime key look-up.
"""

import os
from typing import Literal

from pydantic import BaseModel, Field

PROVIDER_DEFAULT_ENV_VAR: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

LOCAL_ENDPOINT_ENV_VAR: dict[str, str] = {
    "ollama": "OLLAMA_API_BASE",
    "vllm": "VLLM_API_BASE",
    "llama_cpp": "LLAMA_CPP_API_BASE",
}

CLOUD_PROVIDER_API_BASE: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "llama_cpp"})

PROVIDER_AUTH_TEST_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1/auth/key",
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
}

ProviderName = Literal["openrouter", "openai", "anthropic", "google"]
LocalProviderName = Literal["ollama", "vllm", "llama_cpp"]
AnyProviderName = Literal[
    "openrouter", "openai", "anthropic", "google", "ollama", "vllm", "llama_cpp"
]


class APICredential(BaseModel):
    """A named API credential with type classification and key-value fields."""

    credential_name: str = Field(description="Unique credential identifier name")
    credential_type: str = Field(description="Classification type of this credential")
    fields: dict[str, str] = Field(description="Credential key-value data fields")


class LLMProviderConfig(BaseModel):
    """Connection configuration for a single LLM provider endpoint.

    For cloud providers, ``base_url`` is derived from CLOUD_PROVIDER_API_BASE
    at runtime — the stored value acts as an override only. For local providers,
    ``base_url`` is the user-configured endpoint (e.g. http://localhost:11434/v1).
    """

    provider_name: str = Field(description="Provider identifier (e.g. openrouter, ollama)")
    base_url: str = Field(default="", description="Provider API base URL (empty = use known default)")
    api_key_env: str = Field(default="", description="Environment variable name for the API key")
    available_models: list[str] = Field(default_factory=list, description="Available model identifiers")

    @property
    def is_local(self) -> bool:
        return self.provider_name in LOCAL_PROVIDERS


def resolve_provider_api_key(provider: LLMProviderConfig) -> str:
    """Resolve the API key for a provider from the environment.

    For local providers without a configured api_key_env, returns "dummy"
    since local servers typically don't require authentication.
    """
    if provider.api_key_env:
        return os.environ.get(provider.api_key_env, "")
    if provider.is_local:
        return "dummy"
    default_env = PROVIDER_DEFAULT_ENV_VAR.get(provider.provider_name, "")
    if default_env:
        return os.environ.get(default_env, "")
    return ""


def resolve_provider_base_url(provider: LLMProviderConfig) -> str:
    """Resolve the effective base URL for a provider.

    Cloud providers use CLOUD_PROVIDER_API_BASE by default unless overridden.
    Local providers read their URL from the environment variable defined
    in LOCAL_ENDPOINT_ENV_VAR.
    """
    if provider.base_url:
        return provider.base_url

    if provider.is_local:
        env_var = LOCAL_ENDPOINT_ENV_VAR.get(provider.provider_name, "")
        return os.environ.get(env_var, "").strip()

    return CLOUD_PROVIDER_API_BASE.get(provider.provider_name, "")


class AdditionalToolApi(BaseModel):
    """User-configured additional tool API with source selection and key."""

    source_id: str = Field(description="Source identifier from the closed list")
    api_key: str = Field(default="", description="API key for this source")
    base_url: str = Field(default="", description="Override base URL (empty = use default)")


def _default_public_source_ids() -> list[str]:
    """Return all public source IDs so every source is enabled by default."""
    from backend.tools.supports.registry import PUBLIC_DATA_SOURCES
    return [src.source_id for src in PUBLIC_DATA_SOURCES]


class UserSettings(BaseModel):
    """Top-level aggregation of all user configuration."""

    api_credentials: list[APICredential] = Field(
        default_factory=list, description="All configured API credentials"
    )
    llm_providers: list[LLMProviderConfig] = Field(
        default_factory=list, description="All configured LLM providers"
    )
    global_defaults: dict = Field(
        default_factory=dict, description="Global default settings"
    )
    enabled_public_sources: list[str] = Field(
        default_factory=lambda: _default_public_source_ids(),
        description="Source IDs of enabled public data sources",
    )
    additional_tool_apis: list[AdditionalToolApi] = Field(
        default_factory=list, description="User-configured additional tool API entries"
    )

    def get_provider_by_name(self, provider_name: str) -> LLMProviderConfig:
        """Retrieve an LLM provider configuration by its name."""
        for provider in self.llm_providers:
            if provider.provider_name == provider_name:
                return provider
        raise KeyError(
            f"LLM provider '{provider_name}' not found. "
            f"Available providers: {[p.provider_name for p in self.llm_providers]}"
        )

    def get_credential_by_name(self, credential_name: str) -> APICredential:
        """Retrieve an API credential by its unique name."""
        for credential in self.api_credentials:
            if credential.credential_name == credential_name:
                return credential
        raise KeyError(
            f"API credential '{credential_name}' not found. "
            f"Available credentials: {[c.credential_name for c in self.api_credentials]}"
        )

    def get_credential_by_type(self, credential_type: str) -> APICredential:
        """Retrieve the first API credential matching a given type."""
        for credential in self.api_credentials:
            if credential.credential_type == credential_type:
                return credential
        raise KeyError(
            f"API credential with type '{credential_type}' not found. "
            f"Available types: {[c.credential_type for c in self.api_credentials]}"
        )
