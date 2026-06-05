"""
Tool registry for managing tool instances.

What it does:
    Provides a central registry where tool instances are registered by name,
    retrieved by name, listed, and exported as LLM-compatible tool definitions.

Entities in it:
    - ToolRegistry: Singleton-style registry mapping tool names to BaseTool instances.

How used by other modules:
    - The application startup registers all available tools into the registry.
    - The agent execution loop calls get() to resolve tool names from LLM responses.
    - The LLM provider receives get_tool_definitions() output as the tools parameter.
"""

from backend.tools.base import BaseTool


class ToolRegistry:
    """
    Central registry for tool instance management.

    Description:
        Maintains a dictionary mapping tool names to their BaseTool instances.
        Supports registration, retrieval by name, listing, and export of
        tool definitions in the format expected by LLM APIs.

    Attributes:
        _tools: Internal dictionary mapping tool names to BaseTool instances.

    Methods:
        register: Add a tool instance to the registry.
        get: Retrieve a tool by name, raising KeyError if not found.
        list_tools: Return all registered tool names.
        get_tool_definitions: Export tool schemas for LLM API consumption.
    """

    def __init__(self) -> None:
        """
        Initialize an empty tool registry.

        Description:
            Creates the internal storage dictionary for tool instances.

        Params:
            None

        Returns:
            None
        """
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance in the registry.

        Description:
            Adds the tool to the internal dictionary keyed by its name property.

        Params:
            tool (BaseTool): The tool instance to register.

        Returns:
            None
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """
        Retrieve a registered tool by its name.

        Description:
            Looks up and returns the tool instance matching the given name.

        Params:
            name (str): The unique name of the tool to retrieve.

        Returns:
            BaseTool: The tool instance registered under the given name.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found in registry. "
                f"Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """
        Return all registered tool names.

        Description:
            Returns a list of all tool names currently in the registry.

        Params:
            None

        Returns:
            list[str]: List of registered tool names.
        """
        return list(self._tools.keys())

    def get_tool_definitions(self) -> list[dict]:
        """
        Export all tool schemas in the format expected by LLM APIs.

        Description:
            Constructs a list of tool definition dictionaries containing
            the type, function name, description, and parameters schema
            for each registered tool, suitable for passing to LLM chat
            completions as the tools parameter.

        Params:
            None

        Returns:
            list[dict]: List of tool definition objects for LLM API consumption.
        """
        definitions = []
        for tool in self._tools.values():
            definition = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            definitions.append(definition)
        return definitions
