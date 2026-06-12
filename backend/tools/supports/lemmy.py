"""
Lemmy (federated forum) API request builders and response parsers.

What it does:
    Defines request specs and response parsers for the public Lemmy API
    v3.  Covers community post listings and cross-instance search.
    No authentication required for public reads.

Entities in it:
    - BASE_URL: Default Lemmy instance (lemmy.ml).
    - _normalize_community: Strips leading ``!`` from community names.
    - _normalize_sort: Validates sort order against Lemmy's accepted
      values.
    - _normalize_instance: Strips protocol prefix and trailing slash.
    - Request/parse pairs for: posts, search.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://join-lemmy.org/docs/contributors/04-api.html
"""

from typing import Any


BASE_URL = "https://lemmy.ml"

_VALID_SORT_TYPES = frozenset({
    "Hot", "Active", "New", "Old",
    "TopDay", "TopWeek", "TopMonth", "TopAll",
    "MostComments", "NewComments",
})


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_community(raw: str) -> str:
    """Strip a leading ``!`` from a Lemmy community name.

    Args:
        raw: Community name from the LLM (e.g. ``"!cryptocurrency"``).

    Returns:
        Community name without the leading ``!``.
    """
    return raw.strip().lstrip("!")


def _normalize_sort(raw: str) -> str:
    """Validate and return a Lemmy sort type.

    Falls back to ``"Hot"`` when the provided value is not in Lemmy's
    accepted set.

    Args:
        raw: Sort type string from the LLM.

    Returns:
        Validated sort type.
    """
    s = raw.strip()
    if s in _VALID_SORT_TYPES:
        return s
    return "Hot"


def _normalize_instance(raw: str) -> str:
    """Normalize a Lemmy instance identifier to a bare hostname.

    Strips ``https://`` or ``http://`` prefixes and any trailing slashes.

    Args:
        raw: Instance string from the LLM
             (e.g. ``"https://lemmy.world/"``).

    Returns:
        Bare hostname like ``lemmy.world``.
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
        Full instance URL like ``https://lemmy.ml``.
    """
    raw_instance = kwargs.get("instance", "")
    if not raw_instance:
        return BASE_URL
    return f"https://{_normalize_instance(raw_instance)}"


# ---------------------------------------------------------------------------
# posts
# ---------------------------------------------------------------------------

def posts_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for posts from a Lemmy community.

    Args:
        **kwargs: Generic LLM params.  Uses ``community``, ``instance``,
                  ``sort``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    community = _normalize_community(kwargs.get("community", "cryptocurrency"))
    sort = _normalize_sort(kwargs.get("sort", "Hot"))
    limit = min(int(kwargs.get("limit", 20)), 50)
    spec: dict[str, Any] = {
        "path": "/api/v3/post/list",
        "params": {
            "community_name": community,
            "sort": sort,
            "limit": limit,
        },
    }
    instance_url = _instance_base_url(kwargs)
    if instance_url != BASE_URL:
        spec["base_url"] = instance_url
    return spec


def posts_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Lemmy post-list JSON response.

    Args:
        data: Raw JSON dict from the API (contains ``posts`` key).

    Returns:
        List of normalised post dicts with title, body (truncated to
        500 chars), url, ap_id, score, comments, author, published.
    """
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
        for p in data.get("posts", [])
    ]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for searching posts across a Lemmy instance.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``instance``,
                  ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    limit = min(int(kwargs.get("limit", 20)), 50)
    spec: dict[str, Any] = {
        "path": "/api/v3/search",
        "params": {
            "q": query,
            "type_": "Posts",
            "sort": "TopAll",
            "limit": limit,
        },
    }
    instance_url = _instance_base_url(kwargs)
    if instance_url != BASE_URL:
        spec["base_url"] = instance_url
    return spec


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Lemmy search JSON response.

    Args:
        data: Raw JSON dict from the API (contains ``posts`` key).

    Returns:
        List of normalised post dicts with title, body, url, score,
        comments, author, community.
    """
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
        for p in data.get("posts", [])
    ]
