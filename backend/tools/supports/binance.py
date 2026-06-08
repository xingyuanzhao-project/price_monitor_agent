"""
Binance public API support.

Fetches market data from Binance's public REST API v3.
No authentication required.

API base: https://api.binance.com/api/v3/
Docs: https://binance-docs.github.io/apidocs/spot/en/
"""

from typing import Any

import httpx

BASE_URL = "https://api.binance.com/api/v3"


async def fetch_ticker(symbol: str) -> dict[str, Any]:
    """Fetch 24hr ticker price change statistics.

    Args:
        symbol: Binance symbol (e.g. "BTCUSDT").

    Returns:
        Normalized ticker dict with last, high, low, volume, change_pct.
    """
    url = f"{BASE_URL}/ticker/24hr"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"symbol": symbol})
        resp.raise_for_status()
        raw = resp.json()
        return {
            "symbol": raw["symbol"],
            "last": raw["lastPrice"],
            "high": raw["highPrice"],
            "low": raw["lowPrice"],
            "volume": raw["volume"],
            "change_pct": raw["priceChangePercent"],
        }


async def fetch_candlesticks(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch kline/candlestick data.

    Args:
        symbol: Binance symbol (e.g. "BTCUSDT").
        interval: Kline interval (1m, 5m, 1h, 1d, etc).
        limit: Number of candles (max 1000).

    Returns:
        List of normalized candle dicts with ts, o, h, l, c, vol.
    """
    url = f"{BASE_URL}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        candles = []
        for row in resp.json():
            candles.append({
                "ts": row[0], "o": row[1], "h": row[2],
                "l": row[3], "c": row[4], "vol": row[5],
            })
        return candles


async def fetch_orderbook(symbol: str, limit: int = 20) -> dict[str, Any]:
    """Fetch orderbook depth.

    Args:
        symbol: Binance symbol.
        limit: Depth levels (1–5000).

    Returns:
        Dict with "asks" and "bids" lists of [price, qty].
    """
    url = f"{BASE_URL}/depth"
    params = {"symbol": symbol, "limit": min(limit, 5000)}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return {"asks": data["asks"], "bids": data["bids"]}


async def fetch_trades(symbol: str, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch recent trades.

    Args:
        symbol: Binance symbol.
        limit: Number of trades (max 1000).

    Returns:
        List of trade dicts with id, price, qty, time, isBuyerMaker.
    """
    url = f"{BASE_URL}/trades"
    params = {"symbol": symbol, "limit": min(limit, 1000)}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
