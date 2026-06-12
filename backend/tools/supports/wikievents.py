"""
Wikipedia Current Events (Offstream) request builders and response parsers.

What it does:
    Defines request specs and response parsers for the Offstream news API,
    which provides structured current events derived from Wikipedia's Current
    Events Portal.  No authentication required.  No rate limits.
    CC BY-SA 3.0 licensed.

Entities in it:
    - BASE_URL: Offstream news API root.
    - _normalize_query: Strips whitespace from search query strings.
    - _normalize_date_component: Coerces year, month, or day values to
      valid integers.
    - Request/parse pairs for: latest, day.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://offstream.news/api.html
"""

from typing import Any


BASE_URL = "https://offstream.news"


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


def _normalize_date_component(raw: Any, default: int = 1) -> int:
    """Coerce a year, month, or day value to an integer.

    Handles string representations and whitespace that an LLM might
    produce (e.g. ``" 2024 "``, ``"06"``).

    Args:
        raw: Date component from the LLM (int, float, or str).
        default: Fallback when raw is falsy or unparseable.

    Returns:
        Integer date component.
    """
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------

def latest_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for the latest current events.

    Args:
        **kwargs: Generic LLM params.  No required parameters.

    Returns:
        Request spec dict for http.fetch().
    """
    return {"path": "/index.json"}


def latest_parse(data: Any) -> list[dict[str, Any]]:
    """Parse Offstream latest events JSON response.

    Handles multiple response shapes: bare list, dict with ``items`` key,
    or dict with ``news`` key.

    Args:
        data: Raw JSON data from the API (list or dict).

    Returns:
        List of event dicts from the last 3 days.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items", data.get("news", []))
        if isinstance(items, list):
            return items
        return [{"raw": data}]
    return [{"raw": data}]


# ---------------------------------------------------------------------------
# day
# ---------------------------------------------------------------------------

def day_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for current events on a specific date.

    Args:
        **kwargs: Generic LLM params.  Uses ``year``, ``month``, ``day``
                  (coerced to integers via _normalize_date_component).

    Returns:
        Request spec dict for http.fetch().
    """
    year = _normalize_date_component(kwargs.get("year"), 2024)
    month = _normalize_date_component(kwargs.get("month"), 1)
    day = _normalize_date_component(kwargs.get("day"), 1)
    return {"path": f"/news/{year}/{month:02d}/{day}/index.json"}


def day_parse(data: Any) -> list[dict[str, Any]]:
    """Parse Offstream day events JSON response.

    Args:
        data: Raw JSON data from the API (list or dict).

    Returns:
        List of event dicts for the requested date.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("items", data.get("news", [{"raw": data}]))
    return [{"raw": data}]
