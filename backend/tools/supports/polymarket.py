"""
Polymarket Gamma API request builders and response parsers.

What it does:
    Defines request specs and response parsers for Polymarket's Gamma API.
    Covers prediction market listings, events, and keyword search.
    No authentication required. Rate limit: 300 req/10s for markets, 500/10s for events.

Entities in it:
    - BASE_URL: Polymarket Gamma API root.
    - _normalize_tag: Ensures tag is a lowercase slug.
    - Request/parse pairs for: markets, events, search.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://docs.polymarket.com/market-data/overview
"""

from typing import Any


BASE_URL = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_tag(raw: str) -> str:
    """Ensure tag is a lowercase slug as required by Polymarket.

    Args:
        raw: Tag string from the LLM.

    Returns:
        Lowercase stripped tag slug.
    """
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# markets
# ---------------------------------------------------------------------------

def markets_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for active prediction markets sorted by volume.

    Args:
        **kwargs: Generic LLM params.  Uses ``active``, ``limit``, ``tag``,
                  ``order``.

    Returns:
        Request spec dict for http.fetch().
    """
    active = kwargs.get("active", True)
    limit = min(int(kwargs.get("limit", 20)), 100)
    tag = kwargs.get("tag", "")
    order = kwargs.get("order", "volume_24hr")
    params: dict[str, Any] = {
        "active": str(active).lower(),
        "closed": "false",
        "limit": limit,
        "order": order,
        "ascending": "false",
    }
    if tag:
        params["tag_slug"] = _normalize_tag(tag)
    return {"path": "/markets", "params": params, "timeout": 15.0}


def markets_parse(data: list) -> list[dict[str, Any]]:
    """Parse Polymarket markets JSON response.

    Args:
        data: Raw JSON list from the API.

    Returns:
        List of market dicts as returned by the API.
    """
    return data


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def events_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for prediction market events.

    Args:
        **kwargs: Generic LLM params.  Uses ``active``, ``limit``, ``tag``.

    Returns:
        Request spec dict for http.fetch().
    """
    active = kwargs.get("active", True)
    limit = min(int(kwargs.get("limit", 20)), 100)
    tag = kwargs.get("tag", "")
    params: dict[str, Any] = {
        "active": str(active).lower(),
        "closed": "false",
        "limit": limit,
    }
    if tag:
        params["tag_slug"] = _normalize_tag(tag)
    return {"path": "/events", "params": params, "timeout": 15.0}


def events_parse(data: list) -> list[dict[str, Any]]:
    """Parse Polymarket events JSON response.

    Args:
        data: Raw JSON list from the API.

    Returns:
        List of event dicts as returned by the API.
    """
    return data


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec to search prediction markets by keyword.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    limit = min(int(kwargs.get("limit", 20)), 100)
    return {
        "path": "/public-search",
        "params": {"q": query, "limit": limit},
        "timeout": 15.0,
    }


def search_parse(data: Any) -> Any:
    """Parse Polymarket search JSON response.

    Args:
        data: Raw JSON from the API (could be dict or list).

    Returns:
        Search results as returned by the API.
    """
    return data
