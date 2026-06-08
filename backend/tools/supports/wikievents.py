"""
Wikipedia Current Events connector (via Offstream).

Fetches structured current events derived from Wikipedia's Current Events Portal.
No authentication required. No rate limits. CC BY-SA 3.0 licensed.

API base: https://offstream.news
Docs: https://offstream.news/api.html
"""

from typing import Any

import httpx

BASE_URL = "https://offstream.news"


async def fetch_latest() -> list[dict[str, Any]]:
    """Fetch the latest current events from Offstream front page.

    Returns:
        List of event dicts from the last 3 days.
    """
    url = f"{BASE_URL}/index.json"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            items = data.get("items", data.get("news", []))
            if isinstance(items, list):
                return items
            return [{"raw": data}]
        return [{"raw": data}]


async def fetch_day(year: int, month: int, day: int) -> list[dict[str, Any]]:
    """Fetch current events for a specific date.

    Args:
        year: 4-digit year.
        month: Month (1-12).
        day: Day of month.

    Returns:
        List of event dicts for that date.
    """
    url = f"{BASE_URL}/news/{year}/{month:02d}/{day}/index.json"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("items", data.get("news", [{"raw": data}]))
        return [{"raw": data}]
