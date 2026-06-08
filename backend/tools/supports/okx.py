"""
OKX public API connector.

Fetches market data (tickers, candlesticks, orderbook, trades) from OKX's
public v5 REST API. No authentication required for public endpoints.

API base: https://www.okx.com/api/v5/public/
"""

from typing import Any

import httpx

BASE_URL = "https://www.okx.com/api/v5"


async def fetch_ticker(inst_id: str) -> dict[str, Any]:
    """Fetch the latest ticker for an instrument.

    Args:
        inst_id: OKX instrument ID (e.g. "BTC-USDT").

    Returns:
        Ticker data dict from OKX response.
    """
    url = f"{BASE_URL}/market/ticker"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"instId": inst_id})
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
        return data["data"][0] if data.get("data") else {}


async def fetch_candlesticks(
    inst_id: str,
    bar: str = "1H",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch candlestick/OHLCV data for an instrument.

    Args:
        inst_id: OKX instrument ID (e.g. "BTC-USDT").
        bar: Candlestick interval (e.g. "1m", "5m", "1H", "1D").
        limit: Number of candles to return (max 300).

    Returns:
        List of candle dicts with ts, o, h, l, c, vol fields.
    """
    url = f"{BASE_URL}/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": str(min(limit, 300))}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
        candles = []
        for row in data.get("data", []):
            candles.append({
                "ts": row[0], "o": row[1], "h": row[2],
                "l": row[3], "c": row[4], "vol": row[5],
            })
        return candles


async def fetch_orderbook(inst_id: str, depth: int = 20) -> dict[str, Any]:
    """Fetch orderbook for an instrument.

    Args:
        inst_id: OKX instrument ID.
        depth: Number of price levels (max 400).

    Returns:
        Dict with "asks" and "bids" lists.
    """
    url = f"{BASE_URL}/market/books"
    params = {"instId": inst_id, "sz": str(min(depth, 400))}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
        book = data["data"][0] if data.get("data") else {"asks": [], "bids": []}
        return {"asks": book.get("asks", []), "bids": book.get("bids", [])}


async def fetch_trades(inst_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch recent trades for an instrument.

    Args:
        inst_id: OKX instrument ID.
        limit: Number of trades (max 500).

    Returns:
        List of trade dicts.
    """
    url = f"{BASE_URL}/market/trades"
    params = {"instId": inst_id, "limit": str(min(limit, 500))}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', 'unknown')}")
        return data.get("data", [])
