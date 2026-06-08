"""
NewsAPI support.

Fetches global news headlines and articles from NewsAPI.
Requires an API key (free developer tier available).

API base: https://newsapi.org/v2/
Docs: https://newsapi.org/docs
"""

from typing import Any

import httpx

BASE_URL = "https://newsapi.org/v2"


async def fetch_top_headlines(
    api_key: str,
    query: str = "",
    country: str = "us",
    category: str = "",
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """Fetch top headlines.

    Args:
        api_key: NewsAPI API key.
        query: Optional keyword filter.
        country: Country code (e.g. "us", "gb", "de").
        category: Optional category ("business", "technology", "science", etc).
        page_size: Number of results (max 100).

    Returns:
        List of article dicts with title, source, author, url, published_at, description.
    """
    url = f"{BASE_URL}/top-headlines"
    params = {"apiKey": api_key, "country": country, "pageSize": min(page_size, 100)}
    if query:
        params["q"] = query
    if category:
        params["category"] = category
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "author": a.get("author", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "description": a.get("description", ""),
            })
        return articles


async def search_everything(
    api_key: str,
    query: str,
    sort_by: str = "publishedAt",
    language: str = "en",
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """Search all articles.

    Args:
        api_key: NewsAPI API key.
        query: Search query (required).
        sort_by: "relevancy", "popularity", or "publishedAt".
        language: Language code.
        page_size: Number of results (max 100).

    Returns:
        List of article dicts.
    """
    url = f"{BASE_URL}/everything"
    params = {
        "apiKey": api_key,
        "q": query,
        "sortBy": sort_by,
        "language": language,
        "pageSize": min(page_size, 100),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "author": a.get("author", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "description": a.get("description", ""),
            })
        return articles
