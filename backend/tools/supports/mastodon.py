"""
Mastodon public API support.

Fetches hashtag timelines, account search, and (when instance allows)
public timeline data from any Mastodon instance's API.

Note: mastodon.social requires authentication for the public timeline
since 2026. Hashtag timelines and search work without auth.

API base: https://mastodon.social/api/v1/
Docs: https://docs.joinmastodon.org/methods/
"""

from typing import Any

import httpx

DEFAULT_INSTANCE = "https://mastodon.social"


async def fetch_public_timeline(
    instance: str = DEFAULT_INSTANCE,
    limit: int = 20,
    local: bool = False,
) -> list[dict[str, Any]]:
    """Fetch public timeline statuses.

    Args:
        instance: Mastodon instance URL.
        limit: Number of statuses (max 40).
        local: If True, only local statuses.

    Returns:
        List of status dicts with id, content, account, created_at, reblogs_count, favourites_count.
    """
    url = f"{instance}/api/v1/timelines/public"
    params: dict[str, Any] = {"limit": min(limit, 40)}
    if local:
        params["local"] = "true"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        statuses = []
        for s in resp.json():
            statuses.append({
                "id": s["id"],
                "content": s.get("content", ""),
                "account": s.get("account", {}).get("acct", ""),
                "created_at": s.get("created_at", ""),
                "reblogs_count": s.get("reblogs_count", 0),
                "favourites_count": s.get("favourites_count", 0),
                "url": s.get("url", ""),
            })
        return statuses


async def fetch_hashtag_timeline(
    tag: str,
    instance: str = DEFAULT_INSTANCE,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch statuses tagged with a hashtag.

    Args:
        tag: Hashtag (without #).
        instance: Mastodon instance URL.
        limit: Number of statuses.

    Returns:
        List of status dicts.
    """
    url = f"{instance}/api/v1/timelines/tag/{tag}"
    params = {"limit": min(limit, 40)}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        statuses = []
        for s in resp.json():
            statuses.append({
                "id": s["id"],
                "content": s.get("content", ""),
                "account": s.get("account", {}).get("acct", ""),
                "created_at": s.get("created_at", ""),
                "url": s.get("url", ""),
            })
        return statuses


async def search_accounts(
    query: str,
    instance: str = DEFAULT_INSTANCE,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for accounts.

    Args:
        query: Search query.
        instance: Mastodon instance URL.
        limit: Max results.

    Returns:
        List of account dicts with id, acct, display_name, followers_count.
    """
    url = f"{instance}/api/v2/search"
    params = {"q": query, "type": "accounts", "limit": min(limit, 40)}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        accounts = []
        for a in data.get("accounts", []):
            accounts.append({
                "id": a["id"],
                "acct": a["acct"],
                "display_name": a.get("display_name", ""),
                "followers_count": a.get("followers_count", 0),
                "url": a.get("url", ""),
            })
        return accounts
