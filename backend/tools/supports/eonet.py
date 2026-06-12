"""
NASA EONET (Earth Observatory Natural Event Tracker) request builders and
response parsers.

What it does:
    Defines request specs and response parsers for NASA's EONET v3 API.
    Fetches natural events -- wildfires, severe storms, volcanoes, floods,
    sea ice.  No authentication required.

Entities in it:
    - BASE_URL: EONET v3 API root.
    - _normalize_category: Strips whitespace from event category slugs.
    - _normalize_days: Coerces and clamps day-range values to valid integer
      bounds.
    - _normalize_limit: Coerces and clamps result-count limits to valid
      integer bounds.
    - _latest_geometry: Extracts most recent geometry from event array.
    - Request/parse pairs for: events, categories.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://eonet.gsfc.nasa.gov/docs/v3
"""

from typing import Any


BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_category(raw: str) -> str:
    """Strip whitespace from an EONET event category slug.

    EONET categories use identifiers like ``wildfires``,
    ``severeStorms``, ``volcanoes``.  This normalizer trims whitespace
    that an LLM may include around the slug.

    Args:
        raw: Category slug from the LLM.

    Returns:
        Cleaned category string, or empty string if none provided.
    """
    return raw.strip()


def _normalize_days(raw: Any, default: int = 30, lower: int = 1, upper: int = 365) -> int:
    """Coerce a day-range value to an integer within bounds.

    Args:
        raw: Day count from the LLM (int, float, or str).
        default: Fallback when raw is falsy or unparseable.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        Clamped integer day count.
    """
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return default
    return max(lower, min(value, upper))


def _normalize_limit(raw: Any, default: int = 25, lower: int = 1, upper: int = 500) -> int:
    """Coerce a result-count limit to an integer within bounds.

    Args:
        raw: Limit value from the LLM (int, float, or str).
        default: Fallback when raw is falsy or unparseable.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        Clamped integer limit.
    """
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return default
    return max(lower, min(value, upper))


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def events_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for EONET natural events.

    Args:
        **kwargs: Generic LLM params.  Uses ``days``, ``status``, ``limit``,
                  ``category``.

    Returns:
        Request spec dict for http.fetch().
    """
    days = _normalize_days(kwargs.get("days", 30))
    status = kwargs.get("status", "open")
    limit = _normalize_limit(kwargs.get("limit", 25))

    params: dict[str, Any] = {
        "days": days,
        "status": status,
        "limit": limit,
    }
    category = _normalize_category(kwargs.get("category", ""))
    if category:
        params["category"] = category

    return {"path": "/events", "params": params, "timeout": 20.0}


def events_parse(data: dict) -> list[dict[str, Any]]:
    """Parse EONET events JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of event dicts with id, title, categories, sources, geometry,
        closed status.
    """
    events = data.get("events", [])
    return [
        {
            "id": event.get("id", ""),
            "title": event.get("title", ""),
            "categories": [
                category.get("title", "")
                for category in event.get("categories", [])
            ],
            "sources": [
                source.get("url", "")
                for source in event.get("sources", [])
            ],
            "geometry": _latest_geometry(event.get("geometry", [])),
            "closed": event.get("closed"),
        }
        for event in events
    ]


# ---------------------------------------------------------------------------
# categories
# ---------------------------------------------------------------------------

def categories_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for listing EONET event categories.

    Args:
        **kwargs: Generic LLM params.  No required parameters.

    Returns:
        Request spec dict for http.fetch().
    """
    return {"path": "/categories"}


def categories_parse(data: dict) -> list[dict[str, Any]]:
    """Parse EONET categories JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of category dicts with id, title, description.
    """
    return data.get("categories", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_geometry(geometry_list: list) -> dict[str, Any]:
    """Extract most recent geometry point from the geometry array.

    Args:
        geometry_list: List of geometry dicts from an EONET event.

    Returns:
        Dict with date, type, and coordinates of the most recent point.
    """
    if not geometry_list:
        return {}
    latest = geometry_list[-1]
    coordinates = latest.get("coordinates", [])
    return {
        "date": latest.get("date", ""),
        "type": latest.get("type", ""),
        "coordinates": coordinates,
    }
