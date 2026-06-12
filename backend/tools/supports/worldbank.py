"""
World Bank Open Data request builders and response parsers.

What it does:
    Defines request specs and response parsers for the World Bank Indicators
    API v2.  Covers country-level development indicators (GDP, inflation,
    unemployment, etc.) and indicator keyword search across 200+ countries.
    No authentication required.

Entities in it:
    - BASE_URL: World Bank API v2 root.
    - _normalize_country: Lowercases country codes for URL paths.
    - Request/parse pairs for: indicator, search.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581
"""

from typing import Any


BASE_URL = "https://api.worldbank.org/v2"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_country(raw: str) -> str:
    """Lowercase and strip a country code for use in URL paths.

    Accepts 2-letter or 3-letter ISO codes.

    Args:
        raw: Country code from the LLM (e.g. "US", "usa", " GBR ").

    Returns:
        Lowercased, stripped country code.
    """
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# indicator
# ---------------------------------------------------------------------------

def indicator_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a World Bank development indicator.

    Args:
        **kwargs: Generic LLM params.  Uses ``indicator``, ``country``,
                  ``date`` (year range "YYYY:YYYY"), ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    indicator = kwargs.get("indicator", "NY.GDP.MKTP.CD")
    country = _normalize_country(kwargs.get("country", "US"))
    date_range = kwargs.get("date", "2015:2025")
    limit = min(int(kwargs.get("limit", 50)), 500)
    return {
        "path": f"/country/{country}/indicator/{indicator}",
        "params": {
            "format": "json",
            "date": date_range,
            "per_page": limit,
        },
        "timeout": 15.0,
    }


def indicator_parse(data: Any) -> list[dict[str, Any]]:
    """Parse World Bank indicator JSON response.

    The World Bank API returns a two-element list: metadata at index 0
    and records at index 1.

    Args:
        data: Raw JSON (list) from the API.

    Returns:
        List of observation dicts with country, date, value, indicator.
    """
    if not isinstance(data, list) or len(data) < 2:
        return []
    records = data[1] or []
    return [
        {
            "country": r.get("country", {}).get("value", ""),
            "country_code": r.get("countryiso3code", ""),
            "date": r.get("date", ""),
            "value": r.get("value"),
            "indicator": r.get("indicator", {}).get("id", ""),
            "indicator_name": r.get("indicator", {}).get("value", ""),
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for World Bank indicator keyword search.

    Uses the per_page parameter to request only the needed number of
    results.  The query term is included as a path segment when provided
    (World Bank supports ``/indicator?q=...`` for server-side filtering
    via the ``q`` parameter).

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    limit = min(int(kwargs.get("limit", 50)), 1000)
    params: dict[str, Any] = {
        "format": "json",
        "per_page": limit,
        "source": "2",
    }
    if query:
        params["q"] = query
    return {
        "path": "/indicator",
        "params": params,
        "timeout": 30.0,
    }


def search_parse(data: Any) -> list[dict[str, Any]]:
    """Parse World Bank indicator search JSON response.

    Args:
        data: Raw JSON (list) from the API.

    Returns:
        List of indicator dicts with id, name, source.
    """
    if not isinstance(data, list) or len(data) < 2:
        return []
    indicators = data[1] or []
    return [
        {
            "id": ind.get("id", ""),
            "name": ind.get("name", ""),
            "source": ind.get("source", {}).get("value", ""),
        }
        for ind in indicators
    ]
