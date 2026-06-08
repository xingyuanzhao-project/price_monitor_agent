"""
The Guardian Open Platform API support.

Fetches headlines, articles, and search results from The Guardian's free API.
No authentication required for basic access (test key available).

API base: https://content.guardianapis.com/
Docs: https://open-platform.theguardian.com/documentation/
"""

from typing import Any

import httpx

BASE_URL = "https://content.guardianapis.com"


async def search_content(
    query: str,
    api_key: str = "test",
    page_size: int = 20,
    order_by: str = "newest",
) -> list[dict[str, Any]]:
    """Search Guardian content.

    Args:
        query: Search query string.
        api_key: Guardian API key ("test" for development access).
        page_size: Number of results (max 50).
        order_by: "newest", "oldest", or "relevance".

    Returns:
        List of article dicts with id, title, section, date, url.
    """
    url = f"{BASE_URL}/search"
    params = {
        "q": query,
        "api-key": api_key,
        "page-size": min(page_size, 50),
        "order-by": order_by,
        "show-fields": "headline,trailText,byline",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("response", {}).get("results", [])
        return [
            {
                "id": r["id"],
                "title": r.get("webTitle", ""),
                "section": r.get("sectionName", ""),
                "date": r.get("webPublicationDate", ""),
                "url": r.get("webUrl", ""),
                "headline": r.get("fields", {}).get("headline", ""),
                "trail_text": r.get("fields", {}).get("trailText", ""),
            }
            for r in results
        ]


async def fetch_section_headlines(
    section: str = "business",
    api_key: str = "test",
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """Fetch latest headlines from a specific section.

    Args:
        section: Guardian section ID (e.g. "business", "technology", "world").
        api_key: Guardian API key.
        page_size: Number of results.

    Returns:
        List of headline dicts with title, date, url.
    """
    url = f"{BASE_URL}/{section}"
    params = {
        "api-key": api_key,
        "page-size": min(page_size, 50),
        "order-by": "newest",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("response", {}).get("results", [])
        return [
            {
                "title": r.get("webTitle", ""),
                "date": r.get("webPublicationDate", ""),
                "url": r.get("webUrl", ""),
            }
            for r in results
        ]
