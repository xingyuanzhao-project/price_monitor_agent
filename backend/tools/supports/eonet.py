"""
NASA EONET (Earth Observatory Natural Event Tracker) connector.

Fetches natural events -- wildfires, severe storms, volcanoes, floods, sea ice.
No authentication required.

API base: https://eonet.gsfc.nasa.gov/api/v3
Docs: https://eonet.gsfc.nasa.gov/docs/v3
"""

from typing import Any

import httpx

BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"


async def fetch_events(
    days: int = 30,
    status: str = "open",
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Fetch natural events tracked by NASA EONET.

    Args:
        days: Look back window in days (max 365).
        status: "open" (ongoing) or "closed" (resolved) or "all".
        limit: Max events.

    Returns:
        List of event dicts with title, categories, geometry, sources, dates.
    """
    params: dict[str, Any] = {
        "days": min(days, 365),
        "status": status,
        "limit": min(limit, 500),
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{BASE_URL}/events", params=params)
        response.raise_for_status()
        data = response.json()
        events = data.get("events", [])
        return [
            {
                "id": event.get("id", ""),
                "title": event.get("title", ""),
                "categories": [category.get("title", "") for category in event.get("categories", [])],
                "sources": [source.get("url", "") for source in event.get("sources", [])],
                "geometry": _latest_geometry(event.get("geometry", [])),
                "closed": event.get("closed"),
            }
            for event in events
        ]


async def fetch_categories() -> list[dict[str, Any]]:
    """List available EONET event categories.

    Returns:
        List of category dicts with id, title, description.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{BASE_URL}/categories")
        response.raise_for_status()
        data = response.json()
        return data.get("categories", [])


def _latest_geometry(geometry_list: list) -> dict[str, Any]:
    """Extract most recent geometry point from geometry array."""
    if not geometry_list:
        return {}
    latest = geometry_list[-1]
    coordinates = latest.get("coordinates", [])
    return {
        "date": latest.get("date", ""),
        "type": latest.get("type", ""),
        "coordinates": coordinates,
    }
