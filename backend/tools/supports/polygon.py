"""
Polygon.io API support.

Fetches US stock, options, forex, and crypto market data from Polygon.io.
Requires an API key (free tier available).

API base: https://api.polygon.io/
Docs: https://polygon.io/docs
"""

from typing import Any

import httpx

BASE_URL = "https://api.polygon.io"


async def fetch_ticker_details(ticker: str, api_key: str) -> dict[str, Any]:
    """Fetch details about a ticker.

    Args:
        ticker: Stock ticker (e.g. "AAPL").
        api_key: Polygon.io API key.

    Returns:
        Dict with name, market, locale, type, currency, market_cap.
    """
    url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
    params = {"apiKey": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("results", {})
        return {
            "ticker": data.get("ticker"),
            "name": data.get("name"),
            "market": data.get("market"),
            "type": data.get("type"),
            "currency": data.get("currency_name"),
            "market_cap": data.get("market_cap"),
        }


async def fetch_aggregates(
    ticker: str,
    api_key: str,
    multiplier: int = 1,
    timespan: str = "day",
    from_date: str = "",
    to_date: str = "",
    limit: int = 120,
) -> list[dict[str, Any]]:
    """Fetch aggregate bars (OHLCV).

    Args:
        ticker: Stock/crypto/forex ticker.
        api_key: Polygon.io API key.
        multiplier: Size of the timespan multiplier.
        timespan: "minute", "hour", "day", "week", "month".
        from_date: Start date (YYYY-MM-DD).
        to_date: End date (YYYY-MM-DD).
        limit: Max bars (max 50000).

    Returns:
        List of bar dicts with ts, o, h, l, c, v, vw.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"apiKey": api_key, "limit": min(limit, 50000), "sort": "desc"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        bars = []
        for r in data.get("results", []):
            bars.append({
                "ts": r["t"],
                "o": r["o"],
                "h": r["h"],
                "l": r["l"],
                "c": r["c"],
                "v": r["v"],
                "vw": r.get("vw"),
            })
        return bars


async def fetch_last_quote(ticker: str, api_key: str) -> dict[str, Any]:
    """Fetch last NBBO quote for a ticker.

    Args:
        ticker: Stock ticker.
        api_key: Polygon.io API key.

    Returns:
        Dict with bid, ask, bid_size, ask_size, timestamp.
    """
    url = f"{BASE_URL}/v2/last/nbbo/{ticker}"
    params = {"apiKey": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("results", {})
        return {
            "bid": data.get("p"),
            "ask": data.get("P"),
            "bid_size": data.get("s"),
            "ask_size": data.get("S"),
            "timestamp": data.get("t"),
        }
