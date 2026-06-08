"""
Polymarket Gamma API connector.

Fetches prediction market data -- events, markets, probabilities, and volume.
No authentication required. Rate limit: 300 req/10s for markets, 500/10s for events.

API base: https://gamma-api.polymarket.com
Docs: https://docs.polymarket.com/market-data/overview
"""

from typing import Any

import httpx

BASE_URL = "https://gamma-api.polymarket.com"


async def fetch_markets(
    active: bool = True,
    limit: int = 20,
    tag: str = "",
    order: str = "volume_24hr",
) -> list[dict[str, Any]]:
    """Fetch active prediction markets sorted by volume.

    Args:
        active: Only active (open) markets.
        limit: Max markets (max 100).
        tag: Filter by tag slug (e.g. "politics", "crypto", "economics").
        order: Sort field -- volume_24hr, volume, liquidity, competitive.

    Returns:
        List of market dicts with question, outcomePrices, volume, liquidity.
    """
    params: dict[str, Any] = {
        "active": str(active).lower(),
        "closed": "false",
        "limit": min(limit, 100),
        "order": order,
        "ascending": "false",
    }
    if tag:
        params["tag_slug"] = tag
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{BASE_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()


async def fetch_events(
    active: bool = True,
    limit: int = 20,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Fetch prediction market events (top-level questions).

    Args:
        active: Only active events.
        limit: Max events (max 100).
        tag: Filter by tag slug.

    Returns:
        List of event dicts with title, markets, volume.
    """
    params: dict[str, Any] = {
        "active": str(active).lower(),
        "closed": "false",
        "limit": min(limit, 100),
    }
    if tag:
        params["tag_slug"] = tag
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{BASE_URL}/events", params=params)
        response.raise_for_status()
        return response.json()


async def search_markets(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search prediction markets by keyword.

    Args:
        query: Search term (e.g. "fed rate", "bitcoin", "tariff").
        limit: Max results.

    Returns:
        Search results with markets and events.
    """
    params = {"q": query, "limit": min(limit, 100)}
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{BASE_URL}/public-search", params=params)
        response.raise_for_status()
        return response.json()
