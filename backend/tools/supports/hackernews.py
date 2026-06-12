"""
Hacker News (Firebase) API request builders and response parsers.

What it does:
    Defines request specs and response parsers for the public Hacker News
    Firebase API.  Covers top story ID listing and single-item lookup.

Entities in it:
    - BASE_URL: Hacker News Firebase API v0 root.
    - _normalize_item_id: Coerces item IDs from string or numeric input
      with whitespace and float handling.
    - Request/parse pairs for: top_stories, story.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://github.com/HackerNews/API
"""

from typing import Any


BASE_URL = "https://hacker-news.firebaseio.com/v0"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_item_id(raw: Any) -> str:
    """Coerce an item ID to a clean numeric string.

    Handles integer, float, or string input.  Strips surrounding whitespace
    and converts through ``int`` to drop decimal parts from float-like
    strings (e.g. ``"48478969.0"``).

    Args:
        raw: Item ID from the LLM (int, float, or str).

    Returns:
        Numeric string suitable for the HN API path.
    """
    return str(int(float(str(raw).strip())))


# ---------------------------------------------------------------------------
# top_stories (IDs only)
# ---------------------------------------------------------------------------

def top_stories_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for the current top story IDs.

    The HN API returns all ~500 IDs; client-side slicing is applied in
    the parse function.

    Args:
        **kwargs: Generic LLM params.  No required parameters.

    Returns:
        Request spec dict for http.fetch().
    """
    return {"path": "/topstories.json"}


def top_stories_parse(data: list) -> list[int]:
    """Parse Hacker News top-stories JSON response.

    Args:
        data: Raw JSON list of integer story IDs from the API.

    Returns:
        Full list of story IDs (caller applies limit truncation).
    """
    return data


# ---------------------------------------------------------------------------
# story (single item)
# ---------------------------------------------------------------------------

def story_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a single Hacker News item.

    Covers stories, comments, jobs, and polls.

    Args:
        **kwargs: Generic LLM params.  Uses ``item_id`` (coerced to an
                  integer via _normalize_item_id).

    Returns:
        Request spec dict for http.fetch().
    """
    item_id = _normalize_item_id(kwargs.get("item_id", 0))
    return {"path": f"/item/{item_id}.json"}


def story_parse(data: dict | None) -> dict[str, Any]:
    """Parse a single Hacker News item JSON response.

    Args:
        data: Raw JSON dict from the API, or ``None`` for deleted items.

    Returns:
        Normalised item dict with id, type, title, url, text, score,
        by, time, descendants.  Empty dict if the item was deleted.
    """
    if data is None:
        return {}
    return {
        "id": data.get("id"),
        "type": data.get("type"),
        "title": data.get("title", ""),
        "url": data.get("url", ""),
        "text": data.get("text", ""),
        "score": data.get("score", 0),
        "by": data.get("by", ""),
        "time": data.get("time"),
        "descendants": data.get("descendants", 0),
    }
