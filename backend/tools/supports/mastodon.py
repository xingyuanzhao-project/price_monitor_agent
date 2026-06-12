"""
Mastodon public API request builders and response parsers.

What it does:
    Defines request specs and response parsers for the Mastodon REST API
    v1/v2.  Covers public timeline, hashtag timeline, and account search.
    No authentication required for hashtag timelines and search.
    mastodon.social requires authentication for the public timeline
    since 2026.

Entities in it:
    - BASE_URL: Default Mastodon instance (mastodon.social).
    - _normalize_hashtag: Strips leading ``#`` and lowercases.
    - _normalize_instance: Strips protocol prefix and trailing slash.
    - Request/parse pairs for: timeline, hashtag, search.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://docs.joinmastodon.org/methods/
"""

from typing import Any


BASE_URL = "https://mastodon.social"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_hashtag(raw: str) -> str:
    """Strip leading ``#`` and lowercase a hashtag string.

    Args:
        raw: Hashtag string from the LLM, possibly with leading ``#``.

    Returns:
        Lowercase hashtag without the ``#`` prefix.
    """
    return raw.strip().lstrip("#").lower()


def _normalize_instance(raw: str) -> str:
    """Normalize a Mastodon instance identifier to a bare hostname.

    Strips ``https://`` or ``http://`` prefixes and any trailing slashes
    so the result can be used as ``https://{hostname}``.

    Args:
        raw: Instance string from the LLM
             (e.g. ``"https://mastodon.social/"``).

    Returns:
        Bare hostname like ``mastodon.social``.
    """
    s = raw.strip()
    for prefix in ("https://", "http://"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    return s.rstrip("/")


def _instance_base_url(kwargs: dict[str, Any]) -> str:
    """Derive the instance base URL from kwargs, falling back to BASE_URL.

    Args:
        kwargs: Generic LLM params dict.  Uses ``instance``.

    Returns:
        Full instance URL like ``https://mastodon.social``.
    """
    raw_instance = kwargs.get("instance", "")
    if not raw_instance:
        return BASE_URL
    return f"https://{_normalize_instance(raw_instance)}"


# ---------------------------------------------------------------------------
# timeline (public)
# ---------------------------------------------------------------------------

def timeline_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for the public timeline of a Mastodon instance.

    Args:
        **kwargs: Generic LLM params.  Uses ``instance``, ``limit``,
                  ``local``.

    Returns:
        Request spec dict for http.fetch().
    """
    limit = min(int(kwargs.get("limit", 20)), 40)
    params: dict[str, Any] = {"limit": limit}
    if kwargs.get("local"):
        params["local"] = "true"
    spec: dict[str, Any] = {
        "path": "/api/v1/timelines/public",
        "params": params,
        "timeout": 30.0,
    }
    instance_url = _instance_base_url(kwargs)
    if instance_url != BASE_URL:
        spec["base_url"] = instance_url
    return spec


def timeline_parse(data: list) -> list[dict[str, Any]]:
    """Parse Mastodon public timeline JSON response.

    Args:
        data: Raw JSON list of status objects from the API.

    Returns:
        List of normalised status dicts with id, content, account,
        created_at, reblogs_count, favourites_count, url.
    """
    return [
        {
            "id": s["id"],
            "content": s.get("content", ""),
            "account": s.get("account", {}).get("acct", ""),
            "created_at": s.get("created_at", ""),
            "reblogs_count": s.get("reblogs_count", 0),
            "favourites_count": s.get("favourites_count", 0),
            "url": s.get("url", ""),
        }
        for s in data
    ]


# ---------------------------------------------------------------------------
# hashtag
# ---------------------------------------------------------------------------

def hashtag_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for statuses tagged with a specific hashtag.

    Args:
        **kwargs: Generic LLM params.  Uses ``tag`` (or ``query`` as
                  fallback), ``instance``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    tag = _normalize_hashtag(kwargs.get("tag", "") or kwargs.get("query", ""))
    limit = min(int(kwargs.get("limit", 20)), 40)
    spec: dict[str, Any] = {
        "path": f"/api/v1/timelines/tag/{tag}",
        "params": {"limit": limit},
        "timeout": 30.0,
    }
    instance_url = _instance_base_url(kwargs)
    if instance_url != BASE_URL:
        spec["base_url"] = instance_url
    return spec


def hashtag_parse(data: list) -> list[dict[str, Any]]:
    """Parse Mastodon hashtag timeline JSON response.

    Args:
        data: Raw JSON list of status objects from the API.

    Returns:
        List of normalised status dicts with id, content, account,
        created_at, url.
    """
    return [
        {
            "id": s["id"],
            "content": s.get("content", ""),
            "account": s.get("account", {}).get("acct", ""),
            "created_at": s.get("created_at", ""),
            "url": s.get("url", ""),
        }
        for s in data
    ]


# ---------------------------------------------------------------------------
# search (accounts)
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for searching accounts on a Mastodon instance.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``instance``,
                  ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    limit = min(int(kwargs.get("limit", 10)), 40)
    spec: dict[str, Any] = {
        "path": "/api/v2/search",
        "params": {"q": query, "type": "accounts", "limit": limit},
        "timeout": 30.0,
    }
    instance_url = _instance_base_url(kwargs)
    if instance_url != BASE_URL:
        spec["base_url"] = instance_url
    return spec


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Mastodon account-search JSON response.

    Args:
        data: Raw JSON dict from the API (contains ``accounts`` key).

    Returns:
        List of normalised account dicts with id, acct, display_name,
        followers_count, url.
    """
    return [
        {
            "id": a["id"],
            "acct": a["acct"],
            "display_name": a.get("display_name", ""),
            "followers_count": a.get("followers_count", 0),
            "url": a.get("url", ""),
        }
        for a in data.get("accounts", [])
    ]
