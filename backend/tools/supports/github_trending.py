"""
GitHub public API connector for trending/popular repositories.

Uses the GitHub Search API to find recently popular repositories.
No authentication required (rate-limited to 10 requests/minute unauthenticated).

API base: https://api.github.com
Docs: https://docs.github.com/en/rest/search/search
"""

from typing import Any
from datetime import date, timedelta

import httpx

BASE_URL = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "price_monitor_agent/1.0"}


async def fetch_trending(
    language: str = "",
    since_days: int = 7,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch trending repositories by stars gained recently.

    Args:
        language: Filter by programming language (e.g. "python", "rust"). Empty = all.
        since_days: Look back window in days (default 7).
        limit: Max repos to return (max 100).

    Returns:
        List of repo dicts with name, description, stars, language, url.
    """
    cutoff = (date.today() - timedelta(days=since_days)).isoformat()
    q = f"created:>{cutoff}"
    if language:
        q += f" language:{language}"
    url = f"{BASE_URL}/search/repositories"
    params = {"q": q, "sort": "stars", "order": "desc", "per_page": min(limit, 100)}
    async with httpx.AsyncClient(headers=HEADERS, timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "name": r["full_name"],
                "description": r.get("description", ""),
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language", ""),
                "url": r.get("html_url", ""),
                "created_at": r.get("created_at", ""),
                "topics": r.get("topics", []),
            }
            for r in data.get("items", [])
        ]


async def search_repos(
    query: str,
    sort: str = "stars",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search GitHub repositories by keyword.

    Args:
        query: Search term (e.g. "trading bot", "crypto", "quantitative finance").
        sort: Sort by "stars", "forks", "updated", or "help-wanted-issues".
        limit: Max repos.

    Returns:
        List of repo dicts.
    """
    url = f"{BASE_URL}/search/repositories"
    params = {"q": query, "sort": sort, "order": "desc", "per_page": min(limit, 100)}
    async with httpx.AsyncClient(headers=HEADERS, timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "name": r["full_name"],
                "description": r.get("description", ""),
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language", ""),
                "url": r.get("html_url", ""),
                "topics": r.get("topics", []),
            }
            for r in data.get("items", [])
        ]
