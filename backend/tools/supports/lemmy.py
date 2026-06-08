"""
Lemmy (federated forum) public API connector.

Fetches posts and comments from any public Lemmy instance.
No authentication required for public reads. Default instance: lemmy.ml.

API base: {instance}/api/v3
Docs: https://join-lemmy.org/docs/contributors/04-api.html
"""

from typing import Any

import httpx

DEFAULT_INSTANCE = "https://lemmy.ml"


async def fetch_posts(
    community: str = "cryptocurrency",
    instance: str = DEFAULT_INSTANCE,
    sort: str = "Hot",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch posts from a Lemmy community.

    Args:
        community: Community name (e.g. "cryptocurrency", "finance", "wallstreetbets").
        instance: Lemmy instance URL (e.g. "https://lemmy.ml", "https://lemmy.world").
        sort: Sort order -- Hot, Active, New, Old, TopDay, TopWeek, TopMonth, TopAll.
        limit: Max posts (max 50).

    Returns:
        List of post dicts with title, body, url, score, author, comments.
    """
    url = f"{instance}/api/v3/post/list"
    params = {
        "community_name": community,
        "sort": sort,
        "limit": min(limit, 50),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("posts", [])
        return [
            {
                "title": p.get("post", {}).get("name", ""),
                "body": (p.get("post", {}).get("body", "") or "")[:500],
                "url": p.get("post", {}).get("url", ""),
                "ap_id": p.get("post", {}).get("ap_id", ""),
                "score": p.get("counts", {}).get("score", 0),
                "comments": p.get("counts", {}).get("comments", 0),
                "author": p.get("creator", {}).get("name", ""),
                "published": p.get("post", {}).get("published", ""),
            }
            for p in posts
        ]


async def search_posts(
    query: str,
    instance: str = DEFAULT_INSTANCE,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search across a Lemmy instance for posts.

    Args:
        query: Search term.
        instance: Lemmy instance URL.
        limit: Max results.

    Returns:
        List of post dicts.
    """
    url = f"{instance}/api/v3/search"
    params = {
        "q": query,
        "type_": "Posts",
        "sort": "TopAll",
        "limit": min(limit, 50),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("posts", [])
        return [
            {
                "title": p.get("post", {}).get("name", ""),
                "body": (p.get("post", {}).get("body", "") or "")[:500],
                "url": p.get("post", {}).get("url", ""),
                "score": p.get("counts", {}).get("score", 0),
                "comments": p.get("counts", {}).get("comments", 0),
                "author": p.get("creator", {}).get("name", ""),
                "community": p.get("community", {}).get("name", ""),
            }
            for p in posts
        ]
