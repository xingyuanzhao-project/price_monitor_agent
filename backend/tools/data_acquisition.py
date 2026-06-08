"""
Data acquisition tool with dispatch to source-specific support modules.

Receives ToolRequests from the execution harness:
- Native path: source_id + source_type + args → dispatch directly
- Non-native path: text → parse_request extracts source + type + args

Each registered source has a support module in backend/tools/supports/ that
owns the canonical parameter mapping and response normalization.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.tools.base import BaseTool, ToolExecutionError, ToolResult
from backend.tools.supports import okx, binance, coingecko, fred, ecb
from backend.tools.supports import guardian, hackernews, mastodon
from backend.tools.supports import alphavantage, polygon, finnhub, newsapi, twitter, quandl
from backend.tools.supports import yahoo, frankfurter, worldbank, imf
from backend.tools.supports import gdelt, isw, oksurf, wikievents, thehear
from backend.tools.supports import github_trending, lemmy
from backend.tools.supports import polymarket, openmeteo, bis, usgs, gdacs, eonet
from backend.tools.supports import usaspending, comtrade, predscope


# ---------------------------------------------------------------------------
# Dispatch table: source_id → { source_type → async callable }
# Each callable is a function from the corresponding support module.
# ---------------------------------------------------------------------------

DISPATCH: dict[str, dict[str, Any]] = {
    "okx": {
        "ticker": okx.fetch_ticker,
        "ohlcv": okx.fetch_candlesticks,
        "orderbook": okx.fetch_orderbook,
        "trades": okx.fetch_trades,
    },
    "binance": {
        "ticker": binance.fetch_ticker,
        "ohlcv": binance.fetch_candlesticks,
        "orderbook": binance.fetch_orderbook,
        "trades": binance.fetch_trades,
    },
    "coingecko": {
        "ticker": coingecko.fetch_price,
        "ohlcv": coingecko.fetch_market_chart,
        "trending": coingecko.fetch_trending,
    },
    "fred": {
        "series": fred.fetch_series_observations,
        "search": fred.search_series,
        "info": fred.fetch_series_info,
    },
    "ecb": {
        "exchange_rates": ecb.fetch_exchange_rates,
        "interest_rates": ecb.fetch_interest_rates,
    },
    "guardian": {
        "search": guardian.search_content,
        "headlines": guardian.fetch_section_headlines,
    },
    "hackernews": {
        "top_stories": hackernews.fetch_top_stories,
        "top_stories_detail": hackernews.fetch_top_stories_detail,
        "story": hackernews.fetch_item,
    },
    "mastodon": {
        "timeline": mastodon.fetch_public_timeline,
        "hashtag": mastodon.fetch_hashtag_timeline,
        "search": mastodon.search_accounts,
    },
    "alphavantage": {
        "ohlcv": alphavantage.fetch_daily,
        "quote": alphavantage.fetch_quote,
        "crypto": alphavantage.fetch_crypto_exchange_rate,
    },
    "polygon": {
        "ohlcv": polygon.fetch_aggregates,
        "quote": polygon.fetch_last_quote,
        "ticker_details": polygon.fetch_ticker_details,
    },
    "finnhub": {
        "quote": finnhub.fetch_quote,
        "news": finnhub.fetch_company_news,
        "earnings": finnhub.fetch_earnings,
    },
    "newsapi": {
        "headlines": newsapi.fetch_top_headlines,
        "search": newsapi.search_everything,
    },
    "twitter": {
        "search": twitter.search_recent_tweets,
        "timeline": twitter.fetch_user_tweets,
    },
    "quandl": {
        "dataset": quandl.fetch_dataset,
        "metadata": quandl.fetch_dataset_metadata,
    },
    "yahoo": {
        "quote": yahoo.fetch_quote,
        "ohlcv": yahoo.fetch_ohlcv,
    },
    "frankfurter": {
        "latest": frankfurter.fetch_latest,
        "timeseries": frankfurter.fetch_timeseries,
    },
    "worldbank": {
        "indicator": worldbank.fetch_indicator,
        "search": worldbank.search_indicators,
    },
    "imf": {
        "indicator": imf.fetch_indicator,
        "list": imf.list_indicators,
    },
    "gdelt": {
        "search": gdelt.search_articles,
        "timeline": gdelt.fetch_timeline,
    },
    "isw": {
        "latest": isw.fetch_latest,
    },
    "oksurf": {
        "headlines": oksurf.fetch_all_headlines,
        "section": oksurf.fetch_section,
    },
    "wikievents": {
        "latest": wikievents.fetch_latest,
        "day": wikievents.fetch_day,
    },
    "thehear": {
        "country": thehear.fetch_country,
    },
    "github": {
        "trending": github_trending.fetch_trending,
        "search": github_trending.search_repos,
    },
    "lemmy": {
        "posts": lemmy.fetch_posts,
        "search": lemmy.search_posts,
    },
    "polymarket": {
        "markets": polymarket.fetch_markets,
        "events": polymarket.fetch_events,
        "search": polymarket.search_markets,
    },
    "openmeteo": {
        "forecast": openmeteo.fetch_forecast,
        "historical": openmeteo.fetch_historical,
    },
    "bis": {
        "policy_rates": bis.fetch_policy_rates,
        "exchange_rates": bis.fetch_exchange_rates,
    },
    "usgs": {
        "earthquakes": usgs.fetch_earthquakes,
    },
    "gdacs": {
        "events": gdacs.fetch_events,
    },
    "eonet": {
        "events": eonet.fetch_events,
        "categories": eonet.fetch_categories,
    },
    "usaspending": {
        "by_agency": usaspending.fetch_spending_by_agency,
        "over_time": usaspending.fetch_spending_over_time,
    },
    "comtrade": {
        "trade": comtrade.fetch_trade_data,
    },
    "predscope": {
        "markets": predscope.fetch_markets,
        "resolved": predscope.fetch_resolved,
    },
}

# Source name aliases for non-native text parsing
_SOURCE_ALIASES: dict[str, str] = {
    "okx": "okx", "binance": "binance", "coingecko": "coingecko",
    "coin gecko": "coingecko", "fred": "fred", "ecb": "ecb",
    "guardian": "guardian", "the guardian": "guardian",
    "hacker news": "hackernews", "hackernews": "hackernews", "hn": "hackernews",
    "mastodon": "mastodon",
    "alpha vantage": "alphavantage", "alphavantage": "alphavantage",
    "polygon": "polygon", "polygon.io": "polygon",
    "finnhub": "finnhub", "newsapi": "newsapi", "news api": "newsapi",
    "twitter": "twitter", "x": "twitter",
    "quandl": "quandl", "nasdaq": "quandl", "nasdaq data link": "quandl",
    "yahoo": "yahoo", "yahoo finance": "yahoo", "yfinance": "yahoo",
    "frankfurter": "frankfurter", "forex": "frankfurter",
    "world bank": "worldbank", "worldbank": "worldbank",
    "imf": "imf", "international monetary fund": "imf",
    "gdelt": "gdelt",
    "isw": "isw", "understandingwar": "isw", "institute for the study of war": "isw",
    "oksurf": "oksurf", "google news": "oksurf",
    "wikievents": "wikievents", "wikipedia events": "wikievents", "offstream": "wikievents",
    "thehear": "thehear", "the hear": "thehear",
    "github": "github", "github trending": "github",
    "lemmy": "lemmy",
    "polymarket": "polymarket", "prediction market": "polymarket",
    "open-meteo": "openmeteo", "openmeteo": "openmeteo", "open meteo": "openmeteo", "weather": "openmeteo",
    "bis": "bis", "bank for international settlements": "bis",
    "usgs": "usgs", "earthquake": "usgs",
    "gdacs": "gdacs", "disaster alert": "gdacs",
    "eonet": "eonet", "nasa eonet": "eonet", "natural event": "eonet",
    "usaspending": "usaspending", "usa spending": "usaspending", "federal spending": "usaspending",
    "comtrade": "comtrade", "un comtrade": "comtrade", "trade flow": "comtrade",
    "predscope": "predscope",
}

# Type keywords for non-native text parsing
_TYPE_KEYWORDS: dict[str, str] = {
    "candle": "ohlcv", "candlestick": "ohlcv", "ohlcv": "ohlcv", "kline": "ohlcv",
    "price": "ticker", "ticker": "ticker", "quote": "ticker",
    "orderbook": "orderbook", "order book": "orderbook", "depth": "orderbook",
    "trade": "trades", "trades": "trades", "recent trades": "trades",
    "headline": "headlines", "headlines": "headlines", "news": "headlines",
    "article": "search", "search": "search",
    "series": "series", "indicator": "series", "economic": "series",
    "exchange rate": "exchange_rates",
    "subreddit": "subreddit", "timeline": "timeline",
    "top stories": "top_stories", "top": "top_stories",
}

# Regex for symbol extraction
_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{2,10}[-/]?[A-Z]{2,10})\b")


class FetchDataTool(BaseTool):
    """Fetches data from external sources via source-specific support modules."""

    @property
    def name(self) -> str:
        return "fetch_data"

    @property
    def description(self) -> str:
        return (
            "Fetches data from external sources. Specify source_id (e.g. 'okx', "
            "'binance', 'fred') and source_type (e.g. 'ohlcv', 'ticker', 'series') "
            "along with source-specific parameters like symbol or query."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "enum": list(DISPATCH.keys()),
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

    def parse_request(self, text: str) -> dict | None:
        """Parse natural language text into canonical fetch_data arguments.

        Looks for source name keywords, data type keywords, and symbol patterns.
        Returns None if unable to determine at minimum a source_id and source_type.
        """
        text_lower = text.lower()

        source_id = None
        for alias, sid in _SOURCE_ALIASES.items():
            if alias in text_lower:
                source_id = sid
                break

        if source_id is None:
            return None

        source_type = None
        for keyword, stype in _TYPE_KEYWORDS.items():
            if keyword in text_lower:
                if stype in DISPATCH.get(source_id, {}):
                    source_type = stype
                    break

        if source_type is None:
            available_types = list(DISPATCH.get(source_id, {}).keys())
            if available_types:
                source_type = available_types[0]
            else:
                return None

        args: dict[str, Any] = {"source_id": source_id, "source_type": source_type}

        symbol_match = _SYMBOL_PATTERN.search(text)
        if symbol_match:
            args["symbol"] = symbol_match.group(1)

        return args

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the correct support module and return a ToolResult."""
        source_id = kwargs.get("source_id")
        source_type = kwargs.get("source_type")

        if not source_id or source_id not in DISPATCH:
            raise ToolExecutionError(
                f"Unknown source_id: '{source_id}'. Available: {list(DISPATCH.keys())}"
            )

        source_dispatch = DISPATCH[source_id]
        if not source_type or source_type not in source_dispatch:
            raise ToolExecutionError(
                f"Unknown source_type '{source_type}' for source '{source_id}'. "
                f"Available: {list(source_dispatch.keys())}"
            )

        handler_fn = source_dispatch[source_type]

        call_kwargs = {k: v for k, v in kwargs.items() if k not in ("source_id", "source_type") and v is not None}

        # Map generic parameter names to source-specific ones
        call_kwargs = self._map_params(source_id, source_type, call_kwargs)

        try:
            raw_result = await handler_fn(**call_kwargs)
        except TypeError as e:
            raise ToolExecutionError(
                f"Parameter mismatch calling {source_id}.{source_type}: {e}. "
                f"Provided: {list(call_kwargs.keys())}"
            ) from e
        except Exception as e:
            raise ToolExecutionError(
                f"Error fetching from {source_id}/{source_type}: {e}"
            ) from e

        data_type = f"{source_id}_{source_type}"
        size_bytes = len(json.dumps(raw_result, default=str).encode("utf-8"))

        return ToolResult(data_type=data_type, content=raw_result, size_bytes=size_bytes)

    @staticmethod
    def _map_params(source_id: str, source_type: str, params: dict) -> dict:
        """Map generic parameter names to source-specific ones."""
        mapped = dict(params)

        if source_id == "okx":
            if "symbol" in mapped:
                mapped["inst_id"] = mapped.pop("symbol")
            if "interval" in mapped:
                mapped["bar"] = mapped.pop("interval")
            if source_type == "orderbook" and "limit" in mapped:
                mapped["depth"] = mapped.pop("limit")
        elif source_id == "binance" and "symbol" in mapped:
            mapped["symbol"] = mapped["symbol"].replace("-", "")
        elif source_id == "coingecko" and "symbol" in mapped:
            mapped["coin_id"] = mapped.pop("symbol")
        elif source_id == "ecb" and "symbol" in mapped:
            mapped["currency"] = mapped.pop("symbol")

        if source_id == "fred" and "indicator" in mapped:
            mapped["series_id"] = mapped.pop("indicator")

        if source_id == "mastodon" and source_type == "hashtag" and "query" in mapped:
            mapped["tag"] = mapped.pop("query")

        if source_id == "hackernews" and source_type == "story" and "symbol" in mapped:
            mapped["item_id"] = int(mapped.pop("symbol"))

        if source_id == "frankfurter" and "symbol" in mapped:
            mapped["base"] = mapped.pop("symbol")

        return mapped
