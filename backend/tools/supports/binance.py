"""
Binance public API request builders and response parsers.

What it does:
    Defines request specs and response parsers for Binance's public REST API v3.
    Covers 24hr tickers, candlestick/kline data, orderbook depth, and recent trades.
    No authentication required for public endpoints.

Entities in it:
    - BASE_URL: Binance API v3 root.
    - _normalize_symbol: Converts symbol variants to Binance's uppercase
      concatenated format (e.g. "BTC/USDT" -> "BTCUSDT").
    - _normalize_interval: Ensures interval is lowercase.
    - Request/parse pairs for: ticker, ohlcv, orderbook, trades.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://binance-docs.github.io/apidocs/spot/en/
"""

from typing import Any


BASE_URL = "https://api.binance.com/api/v3"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_symbol(raw: str) -> str:
    """Convert any reasonable symbol variant to Binance's concatenated uppercase format.

    Binance expects uppercase concatenated pairs like ``BTCUSDT``.
    Handles ``BTC/USDT``, ``btc-usdt``, ``BTC_USDT``, ``btcusdt``.

    Args:
        raw: Symbol string from the LLM.

    Returns:
        Normalised Binance symbol.
    """
    s = raw.strip().upper()
    s = s.replace("/", "").replace("-", "").replace("_", "")
    return s


def _normalize_interval(raw: str) -> str:
    """Ensure interval string is lowercase as required by Binance.

    Args:
        raw: Interval string from the LLM (e.g. "1H", "1d", "5M").

    Returns:
        Lowercase interval string.
    """
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# ticker
# ---------------------------------------------------------------------------

def ticker_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for 24hr ticker price change statistics.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    return {"path": "/ticker/24hr", "params": {"symbol": symbol}}


def ticker_parse(data: dict) -> dict[str, Any]:
    """Parse Binance 24hr ticker JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Normalized ticker dict with symbol, last, high, low, volume, change_pct.
    """
    return {
        "symbol": data["symbol"],
        "last": data["lastPrice"],
        "high": data["highPrice"],
        "low": data["lowPrice"],
        "volume": data["volume"],
        "change_pct": data["priceChangePercent"],
    }


# ---------------------------------------------------------------------------
# ohlcv (klines/candlesticks)
# ---------------------------------------------------------------------------

def ohlcv_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for kline/candlestick OHLCV data.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``interval``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    interval = _normalize_interval(kwargs.get("interval", "1h"))
    limit = min(int(kwargs.get("limit", 100)), 1000)
    return {
        "path": "/klines",
        "params": {"symbol": symbol, "interval": interval, "limit": limit},
    }


def ohlcv_parse(data: list) -> list[dict[str, Any]]:
    """Parse Binance kline JSON response.

    Args:
        data: Raw JSON list from the API (list of arrays).

    Returns:
        List of candle dicts with ts, o, h, l, c, vol.
    """
    return [
        {"ts": row[0], "o": row[1], "h": row[2], "l": row[3], "c": row[4], "vol": row[5]}
        for row in data
    ]


# ---------------------------------------------------------------------------
# orderbook
# ---------------------------------------------------------------------------

def orderbook_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for orderbook depth.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    limit = min(int(kwargs.get("limit", 20)), 5000)
    return {
        "path": "/depth",
        "params": {"symbol": symbol, "limit": limit},
    }


def orderbook_parse(data: dict) -> dict[str, Any]:
    """Parse Binance orderbook JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with asks and bids lists of [price, qty].
    """
    return {"asks": data["asks"], "bids": data["bids"]}


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------

def trades_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for recent trades.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    limit = min(int(kwargs.get("limit", 100)), 1000)
    return {
        "path": "/trades",
        "params": {"symbol": symbol, "limit": limit},
    }


def trades_parse(data: list) -> list[dict[str, Any]]:
    """Parse Binance recent-trades JSON response.

    Args:
        data: Raw JSON list from the API.

    Returns:
        List of trade dicts.
    """
    return data
