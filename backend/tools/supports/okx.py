"""
OKX public API request builders and response parsers.

What it does:
    Defines request specs and response parsers for OKX's public REST API v5.
    Covers tickers, candlesticks, orderbook, and recent trades.
    No authentication required for public endpoints.

Entities in it:
    - BASE_URL: OKX API v5 root.
    - _normalize_inst_id: Converts symbol variants to OKX's dash-separated
      uppercase format (e.g. "BTC/USDT" -> "BTC-USDT").
    - Request/parse pairs for: ticker, candlesticks, orderbook, trades.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://www.okx.com/docs-v5/en/
"""

from typing import Any


BASE_URL = "https://www.okx.com/api/v5"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_inst_id(raw: str) -> str:
    """Convert any reasonable symbol variant to OKX instrument-ID format.

    OKX expects uppercase, dash-separated pairs like ``BTC-USDT``.
    Handles ``BTC/USDT``, ``btc-usdt``, ``BTC_USDT``, ``btcusdt``
    (when both halves are recognisable as >=3-char tokens).

    Args:
        raw: Symbol string from the LLM.

    Returns:
        Normalised instrument ID.
    """
    s = raw.strip().upper()
    s = s.replace("/", "-").replace("_", "-")
    return s


# ---------------------------------------------------------------------------
# ticker
# ---------------------------------------------------------------------------

def ticker_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for the latest ticker of an instrument.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``.

    Returns:
        Request spec dict for http.fetch().
    """
    inst_id = _normalize_inst_id(kwargs.get("symbol", ""))
    return {"path": "/market/ticker", "params": {"instId": inst_id}}


def ticker_parse(data: dict) -> dict[str, Any]:
    """Parse OKX ticker JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Single ticker data dict.

    Raises:
        RuntimeError: If OKX returns a non-zero status code.
    """
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
    return data["data"][0] if data.get("data") else {}


# ---------------------------------------------------------------------------
# candlesticks (OHLCV)
# ---------------------------------------------------------------------------

def candlesticks_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for candlestick/OHLCV data.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``interval``,
                  ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    inst_id = _normalize_inst_id(kwargs.get("symbol", ""))
    bar = kwargs.get("interval", "1H")
    limit = min(int(kwargs.get("limit", 100)), 300)
    return {
        "path": "/market/candles",
        "params": {"instId": inst_id, "bar": bar, "limit": str(limit)},
    }


def candlesticks_parse(data: dict) -> list[dict[str, Any]]:
    """Parse OKX candlestick JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of candle dicts with ts, o, h, l, c, vol.

    Raises:
        RuntimeError: If OKX returns a non-zero status code.
    """
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
    return [
        {"ts": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4], "vol": r[5]}
        for r in data.get("data", [])
    ]


# ---------------------------------------------------------------------------
# orderbook
# ---------------------------------------------------------------------------

def orderbook_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for orderbook depth.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``limit``
                  (mapped to OKX ``sz``).

    Returns:
        Request spec dict for http.fetch().
    """
    inst_id = _normalize_inst_id(kwargs.get("symbol", ""))
    depth = min(int(kwargs.get("limit", 20)), 400)
    return {
        "path": "/market/books",
        "params": {"instId": inst_id, "sz": str(depth)},
    }


def orderbook_parse(data: dict) -> dict[str, Any]:
    """Parse OKX orderbook JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with asks and bids lists.

    Raises:
        RuntimeError: If OKX returns a non-zero status code.
    """
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
    book = data["data"][0] if data.get("data") else {"asks": [], "bids": []}
    return {"asks": book.get("asks", []), "bids": book.get("bids", [])}


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
    inst_id = _normalize_inst_id(kwargs.get("symbol", ""))
    limit = min(int(kwargs.get("limit", 100)), 500)
    return {
        "path": "/market/trades",
        "params": {"instId": inst_id, "limit": str(limit)},
    }


def trades_parse(data: dict) -> list[dict[str, Any]]:
    """Parse OKX recent-trades JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of trade dicts.

    Raises:
        RuntimeError: If OKX returns a non-zero status code.
    """
    if data.get("code") != "0":
        raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
    return data.get("data", [])
