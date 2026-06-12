"""
Polygon.io API request builders and response parsers.

What it does:
    Defines request specs and response parsers for Polygon.io's REST API.
    Covers aggregate bars (OHLCV), last NBBO quotes, and ticker details.
    Requires a free API key (passed as parameter).

Entities in it:
    - BASE_URL: Polygon.io API root.
    - _normalize_ticker: Uppercases and strips the ticker symbol.
    - Request/parse pairs for: ohlcv, quote, ticker_details.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://polygon.io/docs
"""

from typing import Any


BASE_URL = "https://api.polygon.io"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_ticker(raw: str) -> str:
    """Uppercase and strip the ticker symbol for Polygon.io.

    Args:
        raw: Ticker string from the LLM.

    Returns:
        Uppercase stripped ticker.
    """
    return raw.upper().strip()


# ---------------------------------------------------------------------------
# ohlcv (aggregate bars)
# ---------------------------------------------------------------------------

def ohlcv_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for aggregate bars (OHLCV).

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (as ticker), ``api_key``,
                  ``multiplier``, ``timespan``, ``from_date``, ``to_date``,
                  ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    ticker = _normalize_ticker(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    multiplier = int(kwargs.get("multiplier", 1))
    timespan = kwargs.get("timespan", "day")
    from_date = kwargs.get("from_date", "")
    to_date = kwargs.get("to_date", "")
    limit = min(int(kwargs.get("limit", 120)), 50000)
    return {
        "path": f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
        "params": {"apiKey": api_key, "limit": limit, "sort": "desc"},
    }


def ohlcv_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Polygon.io aggregate bars JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of bar dicts with ts, o, h, l, c, v, vw.
    """
    return [
        {
            "ts": r["t"],
            "o": r["o"],
            "h": r["h"],
            "l": r["l"],
            "c": r["c"],
            "v": r["v"],
            "vw": r.get("vw"),
        }
        for r in data.get("results", [])
    ]


# ---------------------------------------------------------------------------
# quote (last NBBO)
# ---------------------------------------------------------------------------

def quote_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for the last NBBO quote of a ticker.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (as ticker), ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    ticker = _normalize_ticker(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    return {
        "path": f"/v2/last/nbbo/{ticker}",
        "params": {"apiKey": api_key},
    }


def quote_parse(data: dict) -> dict[str, Any]:
    """Parse Polygon.io last NBBO quote JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with bid, ask, bid_size, ask_size, timestamp.
    """
    results = data.get("results", {})
    return {
        "bid": results.get("p"),
        "ask": results.get("P"),
        "bid_size": results.get("s"),
        "ask_size": results.get("S"),
        "timestamp": results.get("t"),
    }


# ---------------------------------------------------------------------------
# ticker_details
# ---------------------------------------------------------------------------

def ticker_details_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for ticker detail information.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (as ticker), ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    ticker = _normalize_ticker(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    return {
        "path": f"/v3/reference/tickers/{ticker}",
        "params": {"apiKey": api_key},
    }


def ticker_details_parse(data: dict) -> dict[str, Any]:
    """Parse Polygon.io ticker details JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with ticker, name, market, type, currency, market_cap.
    """
    results = data.get("results", {})
    return {
        "ticker": results.get("ticker"),
        "name": results.get("name"),
        "market": results.get("market"),
        "type": results.get("type"),
        "currency": results.get("currency_name"),
        "market_cap": results.get("market_cap"),
    }
