"""
Twitter/X API v2 support.

Fetches tweets, timelines, and search results from the Twitter/X API v2.
Requires a Bearer token (developer account).

API base: https://api.twitter.com/2/
Docs: https://developer.twitter.com/en/docs/twitter-api
"""

from typing import Any

import httpx

BASE_URL = "https://api.twitter.com/2"


async def search_recent_tweets(
    query: str,
    bearer_token: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search recent tweets (last 7 days).

    Args:
        query: Twitter search query (supports operators).
        bearer_token: Twitter API Bearer token.
        max_results: Number of results (10-100).

    Returns:
        List of tweet dicts with id, text, created_at, author_id.
    """
    url = f"{BASE_URL}/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "created_at,author_id,public_metrics",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        tweets = []
        for t in data.get("data", []):
            metrics = t.get("public_metrics", {})
            tweets.append({
                "id": t["id"],
                "text": t["text"],
                "created_at": t.get("created_at", ""),
                "author_id": t.get("author_id", ""),
                "retweets": metrics.get("retweet_count", 0),
                "likes": metrics.get("like_count", 0),
                "replies": metrics.get("reply_count", 0),
            })
        return tweets


async def fetch_user_tweets(
    user_id: str,
    bearer_token: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Fetch recent tweets from a user.

    Args:
        user_id: Twitter user ID (numeric).
        bearer_token: Twitter API Bearer token.
        max_results: Number of results (5-100).

    Returns:
        List of tweet dicts.
    """
    url = f"{BASE_URL}/users/{user_id}/tweets"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "max_results": max(5, min(max_results, 100)),
        "tweet.fields": "created_at,public_metrics",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        tweets = []
        for t in data.get("data", []):
            metrics = t.get("public_metrics", {})
            tweets.append({
                "id": t["id"],
                "text": t["text"],
                "created_at": t.get("created_at", ""),
                "retweets": metrics.get("retweet_count", 0),
                "likes": metrics.get("like_count", 0),
            })
        return tweets


async def lookup_user(
    username: str,
    bearer_token: str,
) -> dict[str, Any]:
    """Look up a user by username.

    Args:
        username: Twitter handle (without @).
        bearer_token: Twitter API Bearer token.

    Returns:
        Dict with id, name, username, description, followers_count, tweet_count.
    """
    url = f"{BASE_URL}/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {"user.fields": "description,public_metrics,created_at"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        metrics = data.get("public_metrics", {})
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "username": data.get("username"),
            "description": data.get("description", ""),
            "followers_count": metrics.get("followers_count", 0),
            "tweet_count": metrics.get("tweet_count", 0),
        }
