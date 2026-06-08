"""
Hacker News (Firebase) API support.

Fetches top/new/best stories, story details, and comments from the public
Hacker News API hosted on Firebase.

API base: https://hacker-news.firebaseio.com/v0/
Docs: https://github.com/HackerNews/API
"""

from typing import Any

import httpx

BASE_URL = "https://hacker-news.firebaseio.com/v0"


async def fetch_top_stories(limit: int = 30) -> list[int]:
    """Fetch IDs of current top stories.

    Args:
        limit: Number of story IDs to return (max 500).

    Returns:
        List of story IDs.
    """
    url = f"{BASE_URL}/topstories.json"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        ids = resp.json()
        return ids[:min(limit, 500)]


async def fetch_item(item_id: int) -> dict[str, Any]:
    """Fetch a single item (story, comment, job, poll).

    Args:
        item_id: Hacker News item ID.

    Returns:
        Normalized dict with id, type, title, url, text, score, by, time, descendants.
    """
    url = f"{BASE_URL}/item/{item_id}.json"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        raw = resp.json()
        if raw is None:
            return {}
        return {
            "id": raw.get("id"),
            "type": raw.get("type"),
            "title": raw.get("title", ""),
            "url": raw.get("url", ""),
            "text": raw.get("text", ""),
            "score": raw.get("score", 0),
            "by": raw.get("by", ""),
            "time": raw.get("time"),
            "descendants": raw.get("descendants", 0),
        }


async def fetch_top_stories_detail(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch top stories with full detail.

    Args:
        limit: Number of stories to fetch details for (max 30 to avoid rate limiting).

    Returns:
        List of story dicts.
    """
    ids = await fetch_top_stories(limit=min(limit, 30))
    stories = []
    async with httpx.AsyncClient() as client:
        for story_id in ids:
            url = f"{BASE_URL}/item/{story_id}.json"
            resp = await client.get(url)
            if resp.status_code == 200 and resp.json():
                raw = resp.json()
                stories.append({
                    "id": raw.get("id"),
                    "title": raw.get("title", ""),
                    "url": raw.get("url", ""),
                    "score": raw.get("score", 0),
                    "by": raw.get("by", ""),
                    "time": raw.get("time"),
                    "comments": raw.get("descendants", 0),
                })
    return stories
