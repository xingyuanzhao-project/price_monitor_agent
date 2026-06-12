"""
The Guardian Open Platform request builders and response parsers.

What it does:
    Defines request specs and response parsers for The Guardian's content API.
    Fetches headlines, articles, and search results.  Requires an API key
    (``"test"`` available for development access).

Entities in it:
    - BASE_URL: Guardian content API root.
    - _normalize_section: Lowercases section slug.
    - Request/parse pairs for: search, headlines.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://open-platform.theguardian.com/documentation/
"""

from typing import Any


BASE_URL = "https://content.guardianapis.com"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_section(raw: str) -> str:
    """Lowercase the Guardian section slug.

    Args:
        raw: Section string from the LLM (e.g. "Business", "TECHNOLOGY").

    Returns:
        Lowercased section slug.
    """
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for Guardian content search.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``api_key``,
                  ``limit``, ``order_by``, ``from_date``, ``to_date``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    api_key = kwargs.get("api_key", "test")
    page_size = min(int(kwargs.get("limit", 20)), 50)
    order_by = kwargs.get("order_by", "newest")

    params: dict[str, Any] = {
        "q": query,
        "api-key": api_key,
        "page-size": page_size,
        "order-by": order_by,
        "show-fields": "headline,trailText,byline",
    }
    from_date = kwargs.get("from_date", "")
    if from_date:
        params["from-date"] = from_date
    to_date = kwargs.get("to_date", "")
    if to_date:
        params["to-date"] = to_date

    return {"path": "/search", "params": params}


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Guardian content search JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of article dicts with id, title, section, date, url, headline,
        trail_text.
    """
    results = data.get("response", {}).get("results", [])
    return [
        {
            "id": r["id"],
            "title": r.get("webTitle", ""),
            "section": r.get("sectionName", ""),
            "date": r.get("webPublicationDate", ""),
            "url": r.get("webUrl", ""),
            "headline": r.get("fields", {}).get("headline", ""),
            "trail_text": r.get("fields", {}).get("trailText", ""),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# headlines
# ---------------------------------------------------------------------------

def headlines_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for latest headlines from a Guardian section.

    Args:
        **kwargs: Generic LLM params.  Uses ``section``, ``api_key``,
                  ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    section = _normalize_section(kwargs.get("section", "business"))
    api_key = kwargs.get("api_key", "test")
    page_size = min(int(kwargs.get("limit", 20)), 50)

    return {
        "path": f"/{section}",
        "params": {
            "api-key": api_key,
            "page-size": page_size,
            "order-by": "newest",
        },
    }


def headlines_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Guardian section headlines JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of headline dicts with title, date, url.
    """
    results = data.get("response", {}).get("results", [])
    return [
        {
            "title": r.get("webTitle", ""),
            "date": r.get("webPublicationDate", ""),
            "url": r.get("webUrl", ""),
        }
        for r in results
    ]
