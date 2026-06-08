"""
OKSURF News API connector.

Fetches Google News headlines by section via the free OKSURF REST API.
No authentication required. No rate limits.

API base: https://ok.surf/api/v1
Docs: https://ok.surf/
"""

from typing import Any

import httpx

BASE_URL = "https://ok.surf/api/v1"


async def fetch_all_headlines() -> list[dict[str, Any]]:
    """Fetch all Google News headlines across all sections.

    Returns:
        List of article dicts with title, link, source, section.
    """
    url = f"{BASE_URL}/news-feed"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            articles = []
            for section, items in data.items():
                if isinstance(items, list):
                    for item in items:
                        item["section"] = section
                        articles.append(item)
            return articles
        return data


async def fetch_section(
    section: str = "Business",
) -> list[dict[str, Any]]:
    """Fetch headlines for a specific Google News section.

    Args:
        section: Section name -- US, World, Business, Technology,
                 Entertainment, Sports, Science, Health.

    Returns:
        List of article dicts with title, link, source.
    """
    url = f"{BASE_URL}/news-section"
    payload = {"sections": [section]}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get(section, [])
        return data
