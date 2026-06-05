"""
Abstract base class and error type for all tools in the system.

What it does:
    Defines the interface that all tool implementations must follow, including
    abstract properties for name, description, and parameter schema, and an
    abstract async execute method. Also provides credential injection support.

Entities in it:
    - ToolExecutionError: Exception raised when a tool encounters an error.
    - BaseTool: Abstract base class defining the tool contract.

How used by other modules:
    - All concrete tool classes (data_acquisition, technical_analysis, etc.)
      inherit from BaseTool and implement its abstract interface.
    - The ToolRegistry expects BaseTool instances for registration.
    - The agent execution loop calls inject_credentials() before tool invocation
      and execute() to run the tool with provided parameters.
"""

from abc import ABC, abstractmethod
from typing import Any


class ToolExecutionError(Exception):
    """
    Raised when a tool encounters an error during execution.

    Description:
        Signals that a tool's execute() method failed due to external service
        errors, invalid input, or I/O problems. Carries a descriptive message
        identifying the tool, the operation attempted, and the failure reason.

    Attributes:
        message: Human-readable description of what went wrong.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize with a descriptive error message.

        Description:
            Stores the message and passes it to the base Exception.

        Params:
            message (str): Description of the tool execution failure.

        Returns:
            None
        """
        self.message = message
        super().__init__(message)


class BaseTool(ABC):
    """
    Abstract base class defining the contract for all tools.

    Description:
        Provides the interface for tool registration, description, parameter
        schema exposure, credential injection, and asynchronous execution.
        All concrete tools must implement the abstract properties and methods.

    Attributes:
        _credentials: Dictionary of injected credentials for external service auth.

    Methods:
        name: Abstract property returning the tool's unique name.
        description: Abstract property returning a human-readable description.
        parameters_schema: Abstract property returning the JSON schema for parameters.
        execute: Abstract async method performing the tool's operation.
        inject_credentials: Set credentials for external service authentication.
        credentials: Property to access the injected credentials dictionary.
    """

    def __init__(self) -> None:
        """
        Initialize the base tool with empty credentials.

        Description:
            Sets up the internal credentials dictionary as empty.

        Params:
            None

        Returns:
            None
        """
        self._credentials: dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name identifying this tool for registration and invocation.

        Description:
            Returns the canonical string name used to reference this tool
            in workflow schemas and tool call requests.

        Params:
            None

        Returns:
            str: The unique tool name.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.

        Description:
            Returns a description suitable for display to LLMs and users
            explaining the tool's purpose and capabilities.

        Params:
            None

        Returns:
            str: The tool description.
        """

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """
        JSON Schema defining the parameters this tool accepts.

        Description:
            Returns a dictionary conforming to JSON Schema that describes
            all parameters the tool's execute() method accepts.

        Params:
            None

        Returns:
            dict: JSON Schema for tool parameters.
        """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool's operation with the given parameters.

        Description:
            Performs the tool's primary function asynchronously. Implementations
            must raise ToolExecutionError on any failure rather than returning
            fallback data.

        Params:
            **kwargs (Any): Tool-specific parameters matching parameters_schema.

        Returns:
            Any: The tool's output, structure depends on the specific tool.

        Raises:
            ToolExecutionError: On any execution failure.
        """

    def inject_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Inject authentication credentials for use during execution.

        Description:
            Stores credentials that the tool can access via the credentials
            property when making authenticated external service calls.

        Params:
            credentials (dict[str, Any]): Credential key-value pairs.

        Returns:
            None
        """
        self._credentials = credentials

    @property
    def credentials(self) -> dict[str, Any]:
        """
        Access the injected credentials dictionary.

        Description:
            Returns the credentials previously set via inject_credentials().

        Params:
            None

        Returns:
            dict[str, Any]: The current credentials dictionary.
        """
        return self._credentials
