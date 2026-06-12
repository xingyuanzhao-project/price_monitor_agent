"""
FRED (Federal Reserve Economic Data) request builders and response parsers.

What it does:
    Defines request specs and response parsers for the FRED REST API.
    Covers series observations, series metadata, and keyword search.
    Requires a free API key (register at fredaccount.stlouisfed.org).

Entities in it:
    - BASE_URL: FRED API root.
    - _normalize_series_id: Uppercases series identifiers.
    - Request/parse pairs for: series, search, info.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://fred.stlouisfed.org/docs/api/fred/
"""

from typing import Any


BASE_URL = "https://api.stlouisfed.org/fred"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_series_id(raw: str) -> str:
    """Uppercase and strip a FRED series identifier.

    Args:
        raw: Series ID from the LLM (e.g. "gdp", " CPIAUCSL ").

    Returns:
        Uppercased, stripped series ID.
    """
    return raw.strip().upper()


# ---------------------------------------------------------------------------
# series (observations)
# ---------------------------------------------------------------------------

def series_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for FRED series observations.

    Args:
        **kwargs: Generic LLM params.  Uses ``series_id`` (or ``indicator``),
                  ``api_key``, ``limit``, ``sort_order``.

    Returns:
        Request spec dict for http.fetch().
    """
    series_id = _normalize_series_id(
        kwargs.get("series_id", "") or kwargs.get("indicator", "")
    )
    api_key = kwargs.get("api_key", "")
    limit = int(kwargs.get("limit", 100))
    sort_order = kwargs.get("sort_order", "desc")
    return {
        "path": "/series/observations",
        "params": {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "limit": limit,
            "sort_order": sort_order,
        },
    }


def series_parse(data: dict) -> list[dict[str, Any]]:
    """Parse FRED series observations JSON response.

    Filters out placeholder values (``"."``) that indicate missing data.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of observation dicts with date and value.
    """
    return [
        {"date": obs["date"], "value": obs["value"]}
        for obs in data.get("observations", [])
        if obs.get("value") != "."
    ]


# ---------------------------------------------------------------------------
# info (series metadata)
# ---------------------------------------------------------------------------

def info_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for FRED series metadata.

    Args:
        **kwargs: Generic LLM params.  Uses ``series_id`` (or ``indicator``),
                  ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    series_id = _normalize_series_id(
        kwargs.get("series_id", "") or kwargs.get("indicator", "")
    )
    api_key = kwargs.get("api_key", "")
    return {
        "path": "/series",
        "params": {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        },
    }


def info_parse(data: dict) -> dict[str, Any]:
    """Parse FRED series metadata JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with id, title, frequency, units, seasonal_adjustment.
    """
    series = data.get("seriess", [{}])[0]
    return {
        "id": series.get("id"),
        "title": series.get("title"),
        "frequency": series.get("frequency"),
        "units": series.get("units"),
        "seasonal_adjustment": series.get("seasonal_adjustment"),
    }


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for FRED series keyword search.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``api_key``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    api_key = kwargs.get("api_key", "")
    limit = int(kwargs.get("limit", 20))
    return {
        "path": "/series/search",
        "params": {
            "search_text": query,
            "api_key": api_key,
            "file_type": "json",
            "limit": limit,
        },
    }


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse FRED series search JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of series dicts with id, title, frequency, popularity.
    """
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "frequency": s.get("frequency"),
            "popularity": s.get("popularity"),
        }
        for s in data.get("seriess", [])
    ]
