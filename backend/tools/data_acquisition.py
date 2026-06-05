"""
Data acquisition tool for fetching market, news, social, and macroeconomic data.

What it does:
    Provides a single flexible FetchDataTool that supports 12 source types across
    four categories (market, news, social, macro). The source_type parameter
    selects the specific data endpoint, required parameters, and defaults.

Entities in it:
    - SOURCE_TYPE_DEFINITIONS: Dictionary mapping source type names to their
      category, required parameters, optional parameters, and default values.
    - FetchDataTool: Concrete tool implementation for all data fetching operations.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution to retrieve external data.
    - Credentials are injected by the orchestration layer before execution.
"""

from typing import Any

import httpx

from backend.tools.base import BaseTool, ToolExecutionError


SOURCE_TYPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "market_ohlcv": {
        "category": "market",
        "required_params": ["symbol", "interval"],
        "optional_params": ["start_date", "end_date", "limit"],
        "defaults": {"limit": 100},
    },
    "market_orderbook": {
        "category": "market",
        "required_params": ["symbol"],
        "optional_params": ["depth"],
        "defaults": {"depth": 20},
    },
    "market_trades": {
        "category": "market",
        "required_params": ["symbol"],
        "optional_params": ["limit", "start_date", "end_date"],
        "defaults": {"limit": 100},
    },
    "market_funding": {
        "category": "market",
        "required_params": ["symbol"],
        "optional_params": ["start_date", "end_date", "limit"],
        "defaults": {"limit": 50},
    },
    "news_headlines": {
        "category": "news",
        "required_params": ["query"],
        "optional_params": ["language", "limit", "start_date", "end_date"],
        "defaults": {"language": "en", "limit": 25},
    },
    "news_articles": {
        "category": "news",
        "required_params": ["query"],
        "optional_params": ["language", "limit", "start_date", "end_date", "sources"],
        "defaults": {"language": "en", "limit": 10},
    },
    "news_filings": {
        "category": "news",
        "required_params": ["ticker"],
        "optional_params": ["filing_type", "limit", "start_date", "end_date"],
        "defaults": {"limit": 10},
    },
    "social_posts": {
        "category": "social",
        "required_params": ["query"],
        "optional_params": ["platform", "limit", "start_date", "end_date"],
        "defaults": {"platform": "all", "limit": 50},
    },
    "social_threads": {
        "category": "social",
        "required_params": ["thread_id"],
        "optional_params": ["platform", "include_replies"],
        "defaults": {"include_replies": True},
    },
    "macro_economic": {
        "category": "macro",
        "required_params": ["indicator"],
        "optional_params": ["country", "start_date", "end_date", "frequency"],
        "defaults": {"country": "US", "frequency": "monthly"},
    },
    "macro_onchain": {
        "category": "macro",
        "required_params": ["metric", "network"],
        "optional_params": ["start_date", "end_date", "resolution"],
        "defaults": {"resolution": "daily"},
    },
    "macro_sentiment": {
        "category": "macro",
        "required_params": ["asset"],
        "optional_params": ["metric_type", "start_date", "end_date"],
        "defaults": {"metric_type": "fear_greed"},
    },
}


class FetchDataTool(BaseTool):
    """
    Fetches data from external sources across market, news, social, and macro categories.

    Description:
        A unified data acquisition tool that uses a source_type parameter to
        select the appropriate endpoint and parameter set. Constructs request
        URLs from credential-provided base endpoints and handles authentication
        headers automatically.

    Attributes:
        _credentials: Injected credentials containing source_endpoint and auth_token.

    Methods:
        name: Returns 'fetch_data'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema with source_type and dynamic params.
        execute: Fetches data from the external source with given parameters.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup and LLM tool calls.

        Params:
            None

        Returns:
            str: 'fetch_data'
        """
        return "fetch_data"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains what the tool does and the range of data sources available.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Fetches data from external sources. Supports 12 source types across "
            "market (ohlcv, orderbook, trades, funding), news (headlines, articles, "
            "filings), social (posts, threads), and macro (economic, onchain, sentiment) "
            "categories."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines source_type as a required enum parameter and includes all
            possible optional parameters from all source type definitions.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "enum": list(SOURCE_TYPE_DEFINITIONS.keys()),
                    "description": "The type of data source to fetch from",
                },
                "symbol": {"type": "string", "description": "Trading pair symbol"},
                "interval": {"type": "string", "description": "Candle interval (e.g., '1h', '1d')"},
                "query": {"type": "string", "description": "Search query string"},
                "ticker": {"type": "string", "description": "Stock/asset ticker symbol"},
                "thread_id": {"type": "string", "description": "Social thread identifier"},
                "indicator": {"type": "string", "description": "Economic indicator name"},
                "metric": {"type": "string", "description": "On-chain metric name"},
                "network": {"type": "string", "description": "Blockchain network name"},
                "asset": {"type": "string", "description": "Asset name for sentiment data"},
                "start_date": {"type": "string", "description": "Start date (ISO 8601)"},
                "end_date": {"type": "string", "description": "End date (ISO 8601)"},
                "limit": {"type": "integer", "description": "Maximum records to return"},
                "depth": {"type": "integer", "description": "Orderbook depth levels"},
                "language": {"type": "string", "description": "Language code (e.g., 'en')"},
                "sources": {"type": "string", "description": "Comma-separated source names"},
                "filing_type": {"type": "string", "description": "SEC filing type (e.g., '10-K')"},
                "platform": {"type": "string", "description": "Social platform filter"},
                "include_replies": {"type": "boolean", "description": "Whether to include replies"},
                "country": {"type": "string", "description": "Country code"},
                "frequency": {"type": "string", "description": "Data frequency"},
                "resolution": {"type": "string", "description": "Data resolution"},
                "metric_type": {"type": "string", "description": "Sentiment metric type"},
            },
            "required": ["source_type"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Fetch data from the specified external source.

        Description:
            Validates the source_type, checks required parameters, applies defaults,
            constructs the request URL from credentials, and performs the HTTP request
            with authentication. Returns the response data.

        Params:
            **kwargs (Any): Must include source_type plus all required params for that type.

        Returns:
            dict: The response data from the external source.

        Raises:
            ToolExecutionError: If source_type is invalid, required params are missing,
                credentials are not injected, or the HTTP request fails.
        """
        source_type = kwargs.get("source_type")
        if source_type not in SOURCE_TYPE_DEFINITIONS:
            raise ToolExecutionError(
                f"Unsupported source_type: '{source_type}'. "
                f"Must be one of: {list(SOURCE_TYPE_DEFINITIONS.keys())}"
            )

        definition = SOURCE_TYPE_DEFINITIONS[source_type]
        category = definition["category"]
        required_params = definition["required_params"]
        defaults = definition["defaults"]

        missing_params = [
            param for param in required_params if param not in kwargs or kwargs[param] is None
        ]
        if missing_params:
            raise ToolExecutionError(
                f"Missing required parameters for source_type '{source_type}': {missing_params}"
            )

        source_endpoint = self.credentials.get("source_endpoint")
        if not source_endpoint:
            raise ToolExecutionError(
                "No 'source_endpoint' found in credentials. "
                "Credentials must be injected before calling fetch_data."
            )

        auth_token = self.credentials.get("auth_token")
        if not auth_token:
            raise ToolExecutionError(
                "No 'auth_token' found in credentials. "
                "Credentials must be injected before calling fetch_data."
            )

        request_params = dict(defaults)
        all_known_params = set(required_params) | set(definition["optional_params"])
        for param_name in all_known_params:
            if param_name in kwargs and kwargs[param_name] is not None:
                request_params[param_name] = kwargs[param_name]

        url = f"{source_endpoint.rstrip('/')}/{category}/{source_type}"
        headers = {"Authorization": f"Bearer {auth_token}"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=request_params, headers=headers)
                if response.status_code != 200:
                    raise ToolExecutionError(
                        f"HTTP {response.status_code} from {url}: {response.text}"
                    )
                return response.json()
        except httpx.HTTPError as http_error:
            raise ToolExecutionError(
                f"HTTP request failed for source_type '{source_type}' at {url}: {http_error}"
            ) from http_error
