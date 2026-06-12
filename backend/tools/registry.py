"""
Tool registry for managing tool instances.

What it does:
    Provides a central registry where tool instances are registered by name,
    retrieved by name, listed, and exported as LLM-compatible tool definitions.
    Defines the canonical TOOL_HIERARCHY used by both the backend executor
    and the frontend tool-selector UI.

Entities in it:
    - TOOL_HIERARCHY: Ordered list of tool categories, each with its child
      tool names.  This is the single source of truth consumed by the API
      (``/api/models/tools``) and the frontend hierarchical selector.
    - ToolRegistry: Singleton-style registry mapping tool names to BaseTool instances.

How used by other modules:
    - The application startup registers all available tools into the registry.
    - The agent execution loop calls get() to resolve tool names from LLM responses.
    - The LLM provider receives get_tool_definitions() output as the tools parameter.
    - The models API calls get_tool_hierarchy() to serve category structure.
"""

from backend.tools.base import BaseTool

TOOL_HIERARCHY: list[dict[str, object]] = [
    {
        "category": "Fetch Data",
        "tools": [
            "fetch_exchange_data",
            "fetch_macro_data",
            "fetch_news_data",
            "fetch_social_media_data",
        ],
    },
    {
        "category": "Data Analysis",
        "tools": [
            "technical_analysis",
            "quantitative_analysis",
            "signal_analysis",
            "diagnostic_analysis",
            "detect_regime",
            "estimate_parameters",
            "simulate_process",
            "run_monte_carlo",
        ],
    },
    {
        "category": "Text Analysis",
        "tools": [
            "chunk_text",
            "semantic_search",
            "extract_entities",
            "classify_text",
            "score_text",
            "summarize_text",
            "cross_modal_alignment",
        ],
    },
    {
        "category": "Output",
        "tools": [
            "send_webhook",
            "send_email",
            "send_telegram",
            "write_output",
        ],
    },
]


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

    def get_tool_hierarchy(self) -> list[dict[str, object]]:
        """Return TOOL_HIERARCHY filtered to only contain registered tools.

        Each category entry keeps its order but omits tool names that are
        not present in this registry instance.  Empty categories are dropped.
        """
        registered = set(self._tools)
        result: list[dict[str, object]] = []
        for entry in TOOL_HIERARCHY:
            filtered_tools = [
                t for t in entry["tools"] if t in registered  # type: ignore[union-attr]
            ]
            if filtered_tools:
                result.append({"category": entry["category"], "tools": filtered_tools})
        return result
