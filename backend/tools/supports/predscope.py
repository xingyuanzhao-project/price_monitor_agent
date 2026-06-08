"""
PredScope connector.

Fetches aggregated prediction market data from PredScope's free API.
Covers top 100 active Polymarket markets with probabilities, volume, and outcomes.
No authentication required. Rate limit: 100 req/hour.

API base: https://predscope.com/api
Docs: https://predscope.com/api
"""

from typing import Any

import httpx

BASE_URL = "https://predscope.com/api"


async def fetch_markets() -> dict[str, Any]:
    """Fetch top 100 active prediction markets with outcomes and probabilities.

    Returns:
        Dict with meta (total_markets) and markets list.
        Each market has title, slug, volume, volume_24h, liquidity, categories, outcomes.
        Each outcome has title, probability (0-1), day_change.
    """
    url = f"{BASE_URL}/markets.json"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def fetch_resolved() -> dict[str, Any]:
    """Fetch recently resolved prediction markets with final outcomes.

    Returns:
        Dict with resolved market data -- useful for accuracy research.
    """
    url = f"{BASE_URL}/resolved.json"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
