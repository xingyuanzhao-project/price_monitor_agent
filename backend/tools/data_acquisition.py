"""
Data acquisition tool with dispatch to source-specific support modules.

What it does:
    Routes LLM-generated fetch requests to the correct API endpoint via a
    two-level dispatch table (source_id → source_type → Endpoint).  The
    Endpoint's request function builds an HTTP spec, and http.fetch()
    executes it, traces it, and passes the raw response to the Endpoint's
    parse function.

Entities in it:
    - SOURCE_BASE_URLS: Maps each source_id to its API root URL.
    - DISPATCH: Maps (source_id, source_type) to Endpoint(request, parse).
    - SOURCE_CATEGORIES: Groups source_ids into exchange/macro/news/social.
    - FetchExchangeDataTool, FetchMacroDataTool, FetchNewsDataTool,
      FetchSocialMediaDataTool: Category-specific BaseTool subclasses.
    - _execute_fetch: Shared execution logic called by all four tools.
    - _build_parameters_schema: LLM parameters-schema helper.

How used by other modules:
    - main.py instantiates and registers the four Fetch*DataTool classes.
    - The execution harness dispatches each structured tool call to
      execute(), which delegates to _execute_fetch with the call's
      source_id + source_type + args.
    - _execute_fetch uses http.fetch() as the single HTTP convergence point.
"""

from __future__ import annotations

import json
from typing import Any

from backend.tools.base import BaseTool, ToolExecutionError, ToolResult
from backend.tools.supports.http import Endpoint, fetch as http_fetch
from backend.tools.supports import okx, binance, coingecko, fred, ecb
from backend.tools.supports import guardian, hackernews, mastodon
from backend.tools.supports import alphavantage, polygon, finnhub, newsapi, twitter, quandl
from backend.tools.supports import yahoo, frankfurter, worldbank, imf
from backend.tools.supports import gdelt, isw, oksurf, wikievents, thehear
from backend.tools.supports import github_trending, lemmy
from backend.tools.supports import polymarket, openmeteo, bis, usgs, gdacs, eonet
from backend.tools.supports import usaspending, comtrade, predscope


# ---------------------------------------------------------------------------
# Base URLs — single source of truth per source, owned by the support module
# ---------------------------------------------------------------------------

SOURCE_BASE_URLS: dict[str, str] = {
    "okx": okx.BASE_URL,
    "binance": binance.BASE_URL,
    "coingecko": coingecko.BASE_URL,
    "fred": fred.BASE_URL,
    "ecb": ecb.BASE_URL,
    "guardian": guardian.BASE_URL,
    "hackernews": hackernews.BASE_URL,
    "mastodon": mastodon.BASE_URL,
    "alphavantage": alphavantage.BASE_URL,
    "polygon": polygon.BASE_URL,
    "finnhub": finnhub.BASE_URL,
    "newsapi": newsapi.BASE_URL,
    "twitter": twitter.BASE_URL,
    "quandl": quandl.BASE_URL,
    "yahoo": yahoo.BASE_URL,
    "frankfurter": frankfurter.BASE_URL,
    "worldbank": worldbank.BASE_URL,
    "imf": imf.BASE_URL,
    "gdelt": gdelt.BASE_URL,
    "isw": isw.BASE_URL,
    "oksurf": oksurf.BASE_URL,
    "wikievents": wikievents.BASE_URL,
    "thehear": thehear.BASE_URL,
    "github": github_trending.BASE_URL,
    "lemmy": lemmy.BASE_URL,
    "polymarket": polymarket.BASE_URL,
    "openmeteo": openmeteo.BASE_URL,
    "bis": bis.BASE_URL,
    "usgs": usgs.BASE_URL,
    "gdacs": gdacs.BASE_URL,
    "eonet": eonet.BASE_URL,
    "usaspending": usaspending.BASE_URL,
    "comtrade": comtrade.BASE_URL,
    "predscope": predscope.BASE_URL,
}


# ---------------------------------------------------------------------------
# Dispatch table: source_id → { source_type → Endpoint(request, parse) }
# Each Endpoint pairs a pure request builder with a pure response parser.
# ---------------------------------------------------------------------------

DISPATCH: dict[str, dict[str, Endpoint]] = {
    "okx": {
        "ticker": Endpoint(okx.ticker_request, okx.ticker_parse),
        "ohlcv": Endpoint(okx.candlesticks_request, okx.candlesticks_parse),
        "orderbook": Endpoint(okx.orderbook_request, okx.orderbook_parse),
        "trades": Endpoint(okx.trades_request, okx.trades_parse),
    },
    "binance": {
        "ticker": Endpoint(binance.ticker_request, binance.ticker_parse),
        "ohlcv": Endpoint(binance.ohlcv_request, binance.ohlcv_parse),
        "orderbook": Endpoint(binance.orderbook_request, binance.orderbook_parse),
        "trades": Endpoint(binance.trades_request, binance.trades_parse),
    },
    "coingecko": {
        "ticker": Endpoint(coingecko.ticker_request, coingecko.ticker_parse),
        "ohlcv": Endpoint(coingecko.ohlcv_request, coingecko.ohlcv_parse),
        "trending": Endpoint(coingecko.trending_request, coingecko.trending_parse),
    },
    "fred": {
        "series": Endpoint(fred.series_request, fred.series_parse),
        "search": Endpoint(fred.search_request, fred.search_parse),
        "info": Endpoint(fred.info_request, fred.info_parse),
    },
    "ecb": {
        "exchange_rates": Endpoint(ecb.exchange_rates_request, ecb.exchange_rates_parse),
        "interest_rates": Endpoint(ecb.interest_rates_request, ecb.interest_rates_parse),
    },
    "guardian": {
        "search": Endpoint(guardian.search_request, guardian.search_parse),
        "headlines": Endpoint(guardian.headlines_request, guardian.headlines_parse),
    },
    "hackernews": {
        "top_stories": Endpoint(hackernews.top_stories_request, hackernews.top_stories_parse),
        "story": Endpoint(hackernews.story_request, hackernews.story_parse),
    },
    "mastodon": {
        "timeline": Endpoint(mastodon.timeline_request, mastodon.timeline_parse),
        "hashtag": Endpoint(mastodon.hashtag_request, mastodon.hashtag_parse),
        "search": Endpoint(mastodon.search_request, mastodon.search_parse),
    },
    "alphavantage": {
        "ohlcv": Endpoint(alphavantage.ohlcv_request, alphavantage.ohlcv_parse),
        "quote": Endpoint(alphavantage.quote_request, alphavantage.quote_parse),
        "crypto": Endpoint(alphavantage.crypto_request, alphavantage.crypto_parse),
    },
    "polygon": {
        "ohlcv": Endpoint(polygon.ohlcv_request, polygon.ohlcv_parse),
        "quote": Endpoint(polygon.quote_request, polygon.quote_parse),
        "ticker_details": Endpoint(polygon.ticker_details_request, polygon.ticker_details_parse),
    },
    "finnhub": {
        "quote": Endpoint(finnhub.quote_request, finnhub.quote_parse),
        "news": Endpoint(finnhub.news_request, finnhub.news_parse),
        "earnings": Endpoint(finnhub.earnings_request, finnhub.earnings_parse),
    },
    "newsapi": {
        "headlines": Endpoint(newsapi.headlines_request, newsapi.headlines_parse),
        "search": Endpoint(newsapi.search_request, newsapi.search_parse),
    },
    "twitter": {
        "search": Endpoint(twitter.search_request, twitter.search_parse),
        "timeline": Endpoint(twitter.timeline_request, twitter.timeline_parse),
    },
    "quandl": {
        "dataset": Endpoint(quandl.dataset_request, quandl.dataset_parse),
        "metadata": Endpoint(quandl.metadata_request, quandl.metadata_parse),
    },
    "yahoo": {
        "quote": Endpoint(yahoo.quote_request, yahoo.quote_parse),
        "ohlcv": Endpoint(yahoo.ohlcv_request, yahoo.ohlcv_parse),
    },
    "frankfurter": {
        "latest": Endpoint(frankfurter.latest_request, frankfurter.latest_parse),
        "timeseries": Endpoint(frankfurter.timeseries_request, frankfurter.timeseries_parse),
    },
    "worldbank": {
        "indicator": Endpoint(worldbank.indicator_request, worldbank.indicator_parse),
        "search": Endpoint(worldbank.search_request, worldbank.search_parse),
    },
    "imf": {
        "indicator": Endpoint(imf.indicator_request, imf.indicator_parse),
        "list": Endpoint(imf.list_request, imf.list_parse),
    },
    "gdelt": {
        "search": Endpoint(gdelt.search_request, gdelt.search_parse),
        "timeline": Endpoint(gdelt.timeline_request, gdelt.timeline_parse),
    },
    "isw": {
        "latest": Endpoint(isw.latest_request, isw.latest_parse),
    },
    "oksurf": {
        "headlines": Endpoint(oksurf.headlines_request, oksurf.headlines_parse),
        "section": Endpoint(oksurf.section_request, oksurf.section_parse),
    },
    "wikievents": {
        "latest": Endpoint(wikievents.latest_request, wikievents.latest_parse),
        "day": Endpoint(wikievents.day_request, wikievents.day_parse),
    },
    "thehear": {
        "country": Endpoint(thehear.country_request, thehear.country_parse),
    },
    "github": {
        "trending": Endpoint(github_trending.trending_request, github_trending.trending_parse),
        "search": Endpoint(github_trending.search_request, github_trending.search_parse),
    },
    "lemmy": {
        "posts": Endpoint(lemmy.posts_request, lemmy.posts_parse),
        "search": Endpoint(lemmy.search_request, lemmy.search_parse),
    },
    "polymarket": {
        "markets": Endpoint(polymarket.markets_request, polymarket.markets_parse),
        "events": Endpoint(polymarket.events_request, polymarket.events_parse),
        "search": Endpoint(polymarket.search_request, polymarket.search_parse),
    },
    "openmeteo": {
        "forecast": Endpoint(openmeteo.forecast_request, openmeteo.forecast_parse),
        "historical": Endpoint(openmeteo.historical_request, openmeteo.historical_parse),
    },
    "bis": {
        "policy_rates": Endpoint(bis.policy_rates_request, bis.policy_rates_parse),
        "exchange_rates": Endpoint(bis.exchange_rates_request, bis.exchange_rates_parse),
    },
    "usgs": {
        "earthquakes": Endpoint(usgs.earthquakes_request, usgs.earthquakes_parse),
    },
    "gdacs": {
        "events": Endpoint(gdacs.events_request, gdacs.events_parse),
    },
    "eonet": {
        "events": Endpoint(eonet.events_request, eonet.events_parse),
        "categories": Endpoint(eonet.categories_request, eonet.categories_parse),
    },
    "usaspending": {
        "by_agency": Endpoint(usaspending.by_agency_request, usaspending.by_agency_parse),
        "over_time": Endpoint(usaspending.over_time_request, usaspending.over_time_parse),
    },
    "comtrade": {
        "trade": Endpoint(comtrade.trade_request, comtrade.trade_parse),
    },
    "predscope": {
        "markets": Endpoint(predscope.markets_request, predscope.markets_parse),
        "resolved": Endpoint(predscope.resolved_request, predscope.resolved_parse),
    },
}

SOURCE_CATEGORIES: dict[str, list[str]] = {
    "exchange": [
        "okx", "binance", "coingecko", "yahoo", "alphavantage",
        "polygon", "finnhub", "frankfurter", "polymarket", "predscope",
    ],
    "macro": [
        "fred", "ecb", "worldbank", "imf", "bis",
        "quandl", "usaspending", "comtrade", "openmeteo",
    ],
    "news": [
        "guardian", "hackernews", "newsapi", "gdelt", "isw",
        "oksurf", "wikievents", "thehear", "usgs", "gdacs", "eonet",
    ],
    "social": [
        "mastodon", "twitter", "github", "lemmy",
    ],
}


def _build_parameters_schema(allowed_sources: list[str]) -> dict:
    """Build the LLM parameters schema filtered to *allowed_sources*."""
    return {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "enum": allowed_sources,
                "description": "Which data source to fetch from",
            },
            "source_type": {
                "type": "string",
                "description": "Type of data to fetch (e.g. 'ohlcv', 'ticker', 'search')",
            },
            "symbol": {"type": "string", "description": "Trading pair or asset symbol"},
            "interval": {"type": "string", "description": "Candle interval (e.g. '1h', '1d')"},
            "limit": {"type": "integer", "description": "Max records to return"},
            "query": {"type": "string", "description": "Search query string"},
            "indicator": {"type": "string", "description": "Economic indicator ID"},
        },
        "required": ["source_id", "source_type"],
    }


async def _execute_fetch(
    kwargs: dict[str, Any],
    allowed_sources: list[str],
    enabled_sources: list[str] | None = None,
    emit_event: Any | None = None,
) -> ToolResult:
    """Shared execution logic for all category-specific fetch tools.

    Looks up the Endpoint for (source_id, source_type), calls its request
    builder to produce an HTTP spec, then delegates to http.fetch() which
    makes the call, traces it, and passes the raw response to the parse
    function.
    """
    source_id = kwargs.get("source_id")
    source_type = kwargs.get("source_type")

    if not source_id or source_id not in DISPATCH:
        raise ToolExecutionError(
            f"Unknown source_id: '{source_id}'. Available: {allowed_sources}"
        )
    if source_id not in allowed_sources:
        raise ToolExecutionError(
            f"Source '{source_id}' is not available in this tool. "
            f"Available: {allowed_sources}"
        )
    if enabled_sources is not None and source_id not in enabled_sources:
        raise ToolExecutionError(
            f"Source '{source_id}' is disabled in Settings > Public Data Tools. "
            f"Enable it before fetching."
        )

    source_dispatch = DISPATCH[source_id]
    if not source_type or source_type not in source_dispatch:
        raise ToolExecutionError(
            f"Unknown source_type '{source_type}' for source '{source_id}'. "
            f"Available: {list(source_dispatch.keys())}"
        )

    endpoint = source_dispatch[source_type]
    call_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ("source_id", "source_type") and v is not None
    }

    base_url = SOURCE_BASE_URLS[source_id]
    spec = endpoint.request(**call_kwargs)

    try:
        raw_result = await http_fetch(
            base_url, spec, endpoint.parse, emit_event=emit_event,
        )
    except Exception as exc:
        raise ToolExecutionError(
            f"Error fetching from {source_id}/{source_type}: {exc}"
        ) from exc

    limit = call_kwargs.get("limit")
    if limit is not None and isinstance(raw_result, list):
        raw_result = raw_result[:int(limit)]

    data_type = f"{source_id}_{source_type}"
    size_bytes = len(json.dumps(raw_result, default=str).encode("utf-8"))
    return ToolResult(data_type=data_type, content=raw_result, size_bytes=size_bytes)


# ---------------------------------------------------------------------------
# Category-specific fetch tools
# ---------------------------------------------------------------------------

class FetchExchangeDataTool(BaseTool):
    """Fetches market/exchange data: crypto, stocks, forex, prediction markets."""

    _SOURCES = SOURCE_CATEGORIES["exchange"]

    @property
    def name(self) -> str:
        return "fetch_exchange_data"

    @property
    def description(self) -> str:
        return (
            "Fetches market and exchange data (crypto, stocks, forex, prediction markets). "
            f"Sources: {', '.join(self._SOURCES)}."
        )

    @property
    def parameters_schema(self) -> dict:
        return _build_parameters_schema(self._SOURCES)

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await _execute_fetch(
            kwargs, self._SOURCES, self.credentials.get("_enabled_public_sources"),
            emit_event=self.emit_event,
        )


class FetchMacroDataTool(BaseTool):
    """Fetches macroeconomic data: central bank rates, indicators, trade, weather."""

    _SOURCES = SOURCE_CATEGORIES["macro"]

    @property
    def name(self) -> str:
        return "fetch_macro_data"

    @property
    def description(self) -> str:
        return (
            "Fetches macroeconomic data (central bank rates, economic indicators, "
            f"government spending, trade, climate). Sources: {', '.join(self._SOURCES)}."
        )

    @property
    def parameters_schema(self) -> dict:
        return _build_parameters_schema(self._SOURCES)

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await _execute_fetch(
            kwargs, self._SOURCES, self.credentials.get("_enabled_public_sources"),
            emit_event=self.emit_event,
        )


class FetchNewsDataTool(BaseTool):
    """Fetches news and event data: articles, geopolitical analysis, disaster alerts."""

    _SOURCES = SOURCE_CATEGORIES["news"]

    @property
    def name(self) -> str:
        return "fetch_news_data"

    @property
    def description(self) -> str:
        return (
            "Fetches news and event data (articles, geopolitical analysis, "
            f"disaster alerts, event feeds). Sources: {', '.join(self._SOURCES)}."
        )

    @property
    def parameters_schema(self) -> dict:
        return _build_parameters_schema(self._SOURCES)

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await _execute_fetch(
            kwargs, self._SOURCES, self.credentials.get("_enabled_public_sources"),
            emit_event=self.emit_event,
        )


class FetchSocialMediaDataTool(BaseTool):
    """Fetches social media data: timelines, trending repos, community posts."""

    _SOURCES = SOURCE_CATEGORIES["social"]

    @property
    def name(self) -> str:
        return "fetch_social_media_data"

    @property
    def description(self) -> str:
        return (
            "Fetches social media and community data (timelines, trending repos, "
            f"forum posts). Sources: {', '.join(self._SOURCES)}."
        )

    @property
    def parameters_schema(self) -> dict:
        return _build_parameters_schema(self._SOURCES)

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await _execute_fetch(
            kwargs, self._SOURCES, self.credentials.get("_enabled_public_sources"),
            emit_event=self.emit_event,
        )
