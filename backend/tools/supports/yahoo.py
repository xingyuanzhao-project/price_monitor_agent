"""
Yahoo Finance request builders and response parsers.

What it does:
    Defines request specs and response parsers for Yahoo Finance's chart API.
    Covers real-time quotes and historical OHLCV candlestick data.
    No authentication required. Data is 15-min delayed during US market hours.

Entities in it:
    - BASE_URL: Yahoo Finance chart API root.
    - HEADERS: Custom User-Agent header.
    - _normalize_symbol: Uppercases and strips the ticker symbol.
    - Request/parse pairs for: quote, ohlcv.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API endpoint: https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
"""

from typing import Any


BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {"User-Agent": "price_monitor_agent/1.0"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_symbol(raw: str) -> str:
    """Uppercase and strip the ticker symbol for Yahoo Finance.

    Args:
        raw: Symbol string from the LLM.

    Returns:
        Uppercase stripped symbol.
    """
    return raw.upper().strip()


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------

def quote_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a current stock/ETF/index quote.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    return {
        "path": f"/{symbol}",
        "params": {"interval": "1d", "range": "1d"},
        "headers": HEADERS,
        "timeout": 15.0,
    }


def quote_parse(data: dict) -> dict[str, Any]:
    """Parse Yahoo Finance chart JSON response as a quote.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with symbol, price, previous_close, currency, exchange, market_state.
    """
    result = data["chart"]["result"][0]
    meta = result["meta"]
    return {
        "symbol": meta["symbol"],
        "price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("previousClose"),
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName"),
        "market_state": meta.get("marketState"),
    }


# ---------------------------------------------------------------------------
# ohlcv
# ---------------------------------------------------------------------------

def ohlcv_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for OHLCV candlestick history.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``interval``,
                  ``range_period``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    interval = kwargs.get("interval", "1d")
    range_period = kwargs.get("range_period", "1mo")
    return {
        "path": f"/{symbol}",
        "params": {"interval": interval, "range": range_period},
        "headers": HEADERS,
        "timeout": 15.0,
    }


def ohlcv_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Yahoo Finance chart JSON response as OHLCV candles.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of candle dicts with ts, o, h, l, c, vol.
    """
    result = data["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quotes = result.get("indicators", {}).get("quote", [{}])[0]
    candles = []
    for i, ts in enumerate(timestamps):
        candles.append({
            "ts": ts,
            "o": quotes.get("open", [None])[i],
            "h": quotes.get("high", [None])[i],
            "l": quotes.get("low", [None])[i],
            "c": quotes.get("close", [None])[i],
            "vol": quotes.get("volume", [None])[i],
        })
    return candles
