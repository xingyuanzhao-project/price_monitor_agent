"""
GitHub API request builders and response parsers.

What it does:
    Defines request specs and response parsers for the GitHub REST API.
    Covers trending repositories (by recent stars) and keyword search.
    No authentication required (rate-limited to 10 requests/minute
    unauthenticated).

Entities in it:
    - BASE_URL: GitHub API root.
    - HEADERS: Standard Accept and User-Agent headers for GitHub API.
    - _normalize_language: Lowercases and converts spaces to hyphens.
    - Request/parse pairs for: trending, search.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH
      under the ``"github"`` source_id.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://docs.github.com/en/rest/search/search
"""

from datetime import date, timedelta
from typing import Any


BASE_URL = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "price_monitor_agent/1.0",
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_language(raw: str) -> str:
    """Normalize a programming language name for GitHub search qualifiers.

    Lowercases the input and replaces spaces with hyphens so that
    multi-word language names work correctly (e.g. ``"C Sharp"`` →
    ``"c-sharp"``).  Special characters like ``+`` are preserved
    (e.g. ``"C++"`` → ``"c++"``).

    Args:
        raw: Language name from the LLM.

    Returns:
        Normalised language string suitable for a GitHub ``language:``
        qualifier.
    """
    return raw.strip().lower().replace(" ", "-")


# ---------------------------------------------------------------------------
# trending
# ---------------------------------------------------------------------------

def trending_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for trending repositories by recent stars.

    Uses the GitHub search API with a ``created:>`` date filter, sorted
    by star count descending, to approximate a "trending" list.

    Args:
        **kwargs: Generic LLM params.  Uses ``language``, ``since_days``,
                  ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    since_days = int(kwargs.get("since_days", 7))
    cutoff = (date.today() - timedelta(days=since_days)).isoformat()
    query = f"created:>{cutoff}"

    language = kwargs.get("language", "")
    if language:
        query += f" language:{_normalize_language(language)}"

    limit = min(int(kwargs.get("limit", 20)), 100)
    return {
        "path": "/search/repositories",
        "params": {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        },
        "headers": HEADERS,
    }


def trending_parse(data: dict) -> list[dict[str, Any]]:
    """Parse GitHub search/repositories JSON for trending repos.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of repo dicts with name, description, stars, language, url,
        created_at, topics.
    """
    return [
        {
            "name": r["full_name"],
            "description": r.get("description", ""),
            "stars": r.get("stargazers_count", 0),
            "language": r.get("language", ""),
            "url": r.get("html_url", ""),
            "created_at": r.get("created_at", ""),
            "topics": r.get("topics", []),
        }
        for r in data.get("items", [])
    ]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for searching GitHub repositories by keyword.

    Args:
        **kwargs: Generic LLM params.  Uses ``query``, ``sort``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    query = kwargs.get("query", "")
    sort = kwargs.get("sort", "stars")
    limit = min(int(kwargs.get("limit", 20)), 100)
    return {
        "path": "/search/repositories",
        "params": {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": limit,
        },
        "headers": HEADERS,
    }


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse GitHub search/repositories JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of repo dicts with name, description, stars, language, url,
        topics.
    """
    return [
        {
            "name": r["full_name"],
            "description": r.get("description", ""),
            "stars": r.get("stargazers_count", 0),
            "language": r.get("language", ""),
            "url": r.get("html_url", ""),
            "topics": r.get("topics", []),
        }
        for r in data.get("items", [])
    ]
