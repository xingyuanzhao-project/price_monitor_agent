"""
Pydantic data models for user settings, credentials, and LLM provider configuration.

What it does:
    Defines structured types for API credentials (keyed by name and type),
    LLM provider connection details (endpoint, key, models), and a top-level
    UserSettings container that aggregates all configuration with lookup methods.

Entities in it:
    - APICredential: A named credential with a type classification and arbitrary fields.
    - LLMProviderConfig: Connection configuration for a single LLM provider.
    - UserSettings: Aggregated user settings with credentials, providers, and defaults.

How used by other modules:
    - backend.tools.base injects credentials from UserSettings into tool instances.
    - backend.agent.llm_provider reads LLMProviderConfig to construct provider clients.
    - backend.settings.persistence serializes/deserializes UserSettings to/from JSON.
"""

from pydantic import BaseModel, Field


class APICredential(BaseModel):
    """
    A named API credential with type classification and key-value fields.

    Description:
        Represents a single set of credentials for an external service,
        identified by name and classified by type for lookup purposes.

    Attributes:
        credential_name: Unique name identifying this credential set.
        credential_type: Classification of the credential (e.g., 'api_key', 'oauth').
        fields: Key-value pairs holding the actual credential data.
    """

    credential_name: str = Field(description="Unique credential identifier name")
    credential_type: str = Field(description="Classification type of this credential")
    fields: dict[str, str] = Field(description="Credential key-value data fields")


class LLMProviderConfig(BaseModel):
    """
    Connection configuration for a single LLM provider endpoint.

    Description:
        Stores the endpoint URL, authentication key, and list of available
        models for a specific LLM provider service.

    Attributes:
        provider_name: Unique name identifying this provider.
        base_url: Base URL endpoint for the provider's API.
        api_key: Authentication key for the provider.
        available_models: List of model identifiers available from this provider.
    """

    provider_name: str = Field(description="Unique provider identifier name")
    base_url: str = Field(description="Provider API base URL")
    api_key: str = Field(description="Provider authentication key")
    available_models: list[str] = Field(description="Available model identifiers")


class UserSettings(BaseModel):
    """
    Top-level aggregation of all user configuration.

    Description:
        Combines API credentials, LLM provider configurations, and global
        default settings into a single container. Provides lookup methods
        that raise KeyError when requested items are not found.

    Attributes:
        api_credentials: List of all configured API credentials.
        llm_providers: List of all configured LLM provider connections.
        global_defaults: Dictionary of global default settings.

    Methods:
        get_provider_by_name: Retrieve a provider config by its name.
        get_credential_by_name: Retrieve a credential by its name.
        get_credential_by_type: Retrieve a credential by its type.
    """

    api_credentials: list[APICredential] = Field(
        default_factory=list, description="All configured API credentials"
    )
    llm_providers: list[LLMProviderConfig] = Field(
        default_factory=list, description="All configured LLM providers"
    )
    global_defaults: dict = Field(
        default_factory=dict, description="Global default settings"
    )

    def get_provider_by_name(self, provider_name: str) -> LLMProviderConfig:
        """
        Retrieve an LLM provider configuration by its name.

        Description:
            Searches the llm_providers list for a provider matching the given name.

        Params:
            provider_name (str): The unique name of the provider to find.

        Returns:
            LLMProviderConfig: The matching provider configuration.

        Raises:
            KeyError: If no provider with the given name exists.
        """
        for provider in self.llm_providers:
            if provider.provider_name == provider_name:
                return provider
        raise KeyError(
            f"LLM provider '{provider_name}' not found. "
            f"Available providers: {[p.provider_name for p in self.llm_providers]}"
        )

    def get_credential_by_name(self, credential_name: str) -> APICredential:
        """
        Retrieve an API credential by its unique name.

        Description:
            Searches the api_credentials list for a credential matching the given name.

        Params:
            credential_name (str): The unique name of the credential to find.

        Returns:
            APICredential: The matching credential.

        Raises:
            KeyError: If no credential with the given name exists.
        """
        for credential in self.api_credentials:
            if credential.credential_name == credential_name:
                return credential
        raise KeyError(
            f"API credential '{credential_name}' not found. "
            f"Available credentials: {[c.credential_name for c in self.api_credentials]}"
        )

    def get_credential_by_type(self, credential_type: str) -> APICredential:
        """
        Retrieve the first API credential matching a given type.

        Description:
            Searches the api_credentials list for the first credential with
            the specified type classification.

        Params:
            credential_type (str): The type classification to search for.

        Returns:
            APICredential: The first matching credential.

        Raises:
            KeyError: If no credential with the given type exists.
        """
        for credential in self.api_credentials:
            if credential.credential_type == credential_type:
                return credential
        raise KeyError(
            f"API credential with type '{credential_type}' not found. "
            f"Available types: {[c.credential_type for c in self.api_credentials]}"
        )
