"""
GDELT Project connector.

Fetches global news events from the GDELT DOC 2.0 API. Monitors broadcast,
print, and web news in 100+ languages, updated every 15 minutes.
No authentication required.

API base: https://api.gdeltproject.org/api/v2
Docs: https://www.gdeltproject.org/data.html
"""

from typing import Any

import httpx

BASE_URL = "https://api.gdeltproject.org/api/v2"


async def search_articles(
    query: str,
    mode: str = "artlist",
    limit: int = 25,
    sort: str = "DateDesc",
) -> list[dict[str, Any]]:
    """Search GDELT for news articles matching a query.

    Args:
        query: Search term (e.g. "inflation", "bitcoin crash", "central bank").
        mode: "artlist" (article list) or "tonechart" (tone over time).
        limit: Max articles (max 250).
        sort: "DateDesc", "DateAsc", "ToneDesc", "ToneAsc", "HybridRel".

    Returns:
        List of article dicts with url, title, source, language, date, tone.
    """
    url = f"{BASE_URL}/doc/doc"
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": min(limit, 250),
        "sort": sort,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "url": a.get("url", ""),
                "title": a.get("title", ""),
                "source": a.get("domain", ""),
                "language": a.get("language", ""),
                "seendate": a.get("seendate", ""),
                "tone": a.get("tone", 0),
                "socialimage": a.get("socialimage", ""),
            }
            for a in articles
        ]


async def fetch_timeline(
    query: str,
    mode: str = "timelinevol",
) -> dict[str, Any]:
    """Fetch volume timeline from GDELT DOC API.

    Args:
        query: Search term.
        mode: Timeline mode -- "timelinevol" (volume over time),
              "timelinevolraw" (raw volume), "timelinetone" (tone over time).

    Returns:
        Dict with timeline data (dates and values).
    """
    url = f"{BASE_URL}/doc/doc"
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data
