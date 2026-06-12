"""
GDELT Project request builders and response parsers.

What it does:
    Defines request specs and response parsers for the GDELT DOC 2.0 API.
    Monitors broadcast, print, and web news in 100+ languages, updated every
    15 minutes.  No authentication required.

Entities in it:
    - BASE_URL: GDELT DOC API v2 root.
    - _normalize_mode: Lowercases mode strings to canonical GDELT format.
    - Request/parse pairs for: search, timeline.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://www.gdeltproject.org/data.html
"""

from typing import Any


BASE_URL = "https://api.gdeltproject.org/api/v2"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_mode(raw: str) -> str:
    """Lowercase the GDELT mode string to its canonical form.

    GDELT expects lowercase mode values like ``artlist``, ``tonechart``,
    ``timelinevol``.

    Args:
        raw: Mode string from the LLM.

    Returns:
        Lowercased mode string.
    """
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for GDELT article search.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``mode``, ``limit``,
                  ``sort``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    mode = _normalize_mode(kwargs.get("mode", "artlist"))
    limit = min(int(kwargs.get("limit", 25)), 250)
    sort = kwargs.get("sort", "DateDesc")
    return {
        "path": "/doc/doc",
        "params": {
            "query": query,
            "mode": mode,
            "format": "json",
            "maxrecords": limit,
            "sort": sort,
        },
        "timeout": 20.0,
    }


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse GDELT article search JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of article dicts with url, title, source, language, date, tone.
    """
    articles = data.get("articles", [])
    return [
        {
            "url": a.get("url", ""),
            "title": a.get("title", ""),
            "source": a.get("domain", ""),
            "language": a.get("language", ""),
            "seendate": a.get("seendate", ""),
            "tone": a.get("tone", 0),
            "socialimage": a.get("socialimage", ""),
        }
        for a in articles
    ]


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------

def timeline_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for GDELT volume/tone timeline.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``mode``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    mode = _normalize_mode(kwargs.get("mode", "timelinevol"))
    return {
        "path": "/doc/doc",
        "params": {
            "query": query,
            "mode": mode,
            "format": "json",
        },
        "timeout": 20.0,
    }


def timeline_parse(data: dict) -> dict[str, Any]:
    """Parse GDELT timeline JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with timeline data (dates and values).
    """
    return data
