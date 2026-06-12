"""
Twitter/X API v2 request builders and response parsers.

What it does:
    Defines request specs and response parsers for the Twitter/X REST
    API v2.  Covers recent-tweet search and user-timeline retrieval.
    Requires a Bearer token passed via ``api_key`` in kwargs.

Entities in it:
    - BASE_URL: Twitter API v2 root.
    - _normalize_query: Strips whitespace from search query strings.
    - _normalize_user_id: Strips leading @ sign and whitespace from user
      IDs.
    - Request/parse pairs for: search, timeline.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://developer.twitter.com/en/docs/twitter-api
"""

from typing import Any


BASE_URL = "https://api.twitter.com/2"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_query(raw: str) -> str:
    """Strip whitespace from a search query string.

    Args:
        raw: Query string from the LLM.

    Returns:
        Trimmed query string.
    """
    return raw.strip()


def _normalize_user_id(raw: str) -> str:
    """Strip leading @ sign and surrounding whitespace from a user ID.

    LLMs sometimes include the @ prefix or trailing spaces when
    referencing Twitter user IDs.

    Args:
        raw: User ID string from the LLM.

    Returns:
        Clean user ID string.
    """
    return raw.strip().lstrip("@")


# ---------------------------------------------------------------------------
# search (recent tweets)
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for searching recent tweets (last 7 days).

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``api_key``
                  (Bearer token), ``max_results`` or ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    bearer_token = kwargs.get("api_key", "") or kwargs.get("bearer_token", "")
    query = _normalize_query(kwargs.get("query", ""))
    max_results = max(10, min(
        int(kwargs.get("max_results", 0) or kwargs.get("limit", 10)),
        100,
    ))
    return {
        "path": "/tweets/search/recent",
        "params": {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,author_id,public_metrics",
        },
        "headers": {"Authorization": f"Bearer {bearer_token}"},
    }


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Twitter recent-tweet search JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of tweet dicts with id, text, created_at, author_id,
        retweets, likes, replies.
    """
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


# ---------------------------------------------------------------------------
# timeline (user tweets)
# ---------------------------------------------------------------------------

def timeline_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for recent tweets from a specific user.

    Args:
        **kwargs: Generic LLM params.  Uses ``user_id``, ``api_key``
                  (Bearer token), ``max_results`` or ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    bearer_token = kwargs.get("api_key", "") or kwargs.get("bearer_token", "")
    user_id = _normalize_user_id(kwargs.get("user_id", ""))
    max_results = max(5, min(
        int(kwargs.get("max_results", 0) or kwargs.get("limit", 10)),
        100,
    ))
    return {
        "path": f"/users/{user_id}/tweets",
        "params": {
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics",
        },
        "headers": {"Authorization": f"Bearer {bearer_token}"},
    }


def timeline_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Twitter user-timeline JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of tweet dicts with id, text, created_at, retweets, likes.
    """
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
