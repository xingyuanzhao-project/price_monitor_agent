"""
OKSURF News API request builders and response parsers.

What it does:
    Defines request specs and response parsers for the OKSURF Google News API.
    Covers all-headlines feed and section-specific headline retrieval.
    No authentication required. No rate limits.

Entities in it:
    - BASE_URL: OKSURF API v1 root.
    - _normalize_section: Converts section name to Title Case.
    - Request/parse pairs for: headlines, section.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://ok.surf/
"""

from typing import Any


BASE_URL = "https://ok.surf/api/v1"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_section(raw: str) -> str:
    """Convert section name to Title Case as expected by OKSURF.

    Valid sections: US, World, Business, Technology, Entertainment,
    Sports, Science, Health.

    Args:
        raw: Section string from the LLM.

    Returns:
        Title-cased section name.
    """
    return raw.strip().title()


# ---------------------------------------------------------------------------
# headlines (all sections)
# ---------------------------------------------------------------------------

def headlines_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for all Google News headlines across all sections.

    Args:
        **kwargs: Generic LLM params.  No params needed for this endpoint.

    Returns:
        Request spec dict for http.fetch().
    """
    return {"path": "/news-feed", "params": {}, "timeout": 15.0}


def headlines_parse(data: Any) -> list[dict[str, Any]]:
    """Parse OKSURF all-headlines JSON response.

    Args:
        data: Raw JSON from the API (dict keyed by section, or list).

    Returns:
        List of article dicts with title, link, source, section.
    """
    if isinstance(data, dict):
        articles = []
        for section, items in data.items():
            if isinstance(items, list):
                for item in items:
                    item["section"] = section
                    articles.append(item)
        return articles
    return data


# ---------------------------------------------------------------------------
# section (specific section via POST)
# ---------------------------------------------------------------------------

def section_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for headlines from a specific Google News section.

    Args:
        **kwargs: Generic LLM params.  Uses ``section``.

    Returns:
        Request spec dict for http.fetch() with POST method.
    """
    section = _normalize_section(kwargs.get("section", "Business"))
    return {
        "path": "/news-section",
        "params": {},
        "method": "POST",
        "body": {"sections": [section]},
        "timeout": 15.0,
    }


def section_parse(data: Any) -> list[dict[str, Any]]:
    """Parse OKSURF section headlines JSON response.

    Args:
        data: Raw JSON from the API (dict keyed by section name, or list).

    Returns:
        List of article dicts with title, link, source.
    """
    if isinstance(data, dict):
        section_name = next(iter(data), "")
        return data.get(section_name, [])
    return data
