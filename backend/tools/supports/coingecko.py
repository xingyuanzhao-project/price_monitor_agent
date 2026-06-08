"""
CoinGecko public API support.

Fetches aggregated crypto market data from CoinGecko's free API.
No authentication required for public endpoints.

API base: https://api.coingecko.com/api/v3/
Docs: https://docs.coingecko.com/reference/introduction
"""

from typing import Any

import httpx

BASE_URL = "https://api.coingecko.com/api/v3"


async def fetch_price(coin_id: str, vs_currency: str = "usd") -> dict[str, Any]:
    """Fetch simple price for a coin.

    Args:
        coin_id: CoinGecko coin ID (e.g. "bitcoin", "ethereum").
        vs_currency: Target currency (e.g. "usd", "eur").

    Returns:
        Dict with coin_id, price, market_cap, vol_24h, change_24h.
    """
    url = f"{BASE_URL}/simple/price"
    params = {
        "ids": coin_id,
        "vs_currencies": vs_currency,
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get(coin_id, {})
        return {
            "coin_id": coin_id,
            "price": data.get(vs_currency),
            "market_cap": data.get(f"{vs_currency}_market_cap"),
            "vol_24h": data.get(f"{vs_currency}_24h_vol"),
            "change_24h": data.get(f"{vs_currency}_24h_change"),
        }


async def fetch_market_chart(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 7,
) -> dict[str, list]:
    """Fetch historical market chart data.

    Args:
        coin_id: CoinGecko coin ID.
        vs_currency: Target currency.
        days: Number of days of history (1, 7, 14, 30, 90, 180, 365, max).

    Returns:
        Dict with "prices", "market_caps", "total_volumes" — each a list of [ts, value].
    """
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": str(days)}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return {
            "prices": data.get("prices", []),
            "market_caps": data.get("market_caps", []),
            "total_volumes": data.get("total_volumes", []),
        }


async def fetch_trending() -> list[dict[str, Any]]:
    """Fetch trending coins.

    Returns:
        List of trending coin dicts with id, name, symbol, market_cap_rank.
    """
    url = f"{BASE_URL}/search/trending"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        coins = resp.json().get("coins", [])
        return [
            {
                "id": c["item"]["id"],
                "name": c["item"]["name"],
                "symbol": c["item"]["symbol"],
                "market_cap_rank": c["item"].get("market_cap_rank"),
            }
            for c in coins
        ]
