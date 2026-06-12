"""
NewsAPI request builders and response parsers.

What it does:
    Defines request specs and response parsers for the NewsAPI REST API.
    Fetches global news headlines and articles.  Requires an API key
    (free developer tier available).

Entities in it:
    - BASE_URL: NewsAPI v2 root.
    - _normalize_language: Ensures 2-letter lowercase language codes.
    - _normalize_sort_by: Ensures snake_case sort parameter.
    - Request/parse pairs for: headlines, search.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://newsapi.org/docs
"""

from typing import Any


BASE_URL = "https://newsapi.org/v2"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_language(raw: str) -> str:
    """Normalize language to 2-letter lowercase ISO code.

    Args:
        raw: Language string from the LLM (e.g. "EN", "English", "en").

    Returns:
        Lowercased 2-letter language code.
    """
    return raw.strip().lower()[:2]


def _normalize_sort_by(raw: str) -> str:
    """Normalize sort parameter to NewsAPI's expected camelCase values.

    NewsAPI accepts ``relevancy``, ``popularity``, ``publishedAt``.
    Converts common variants like ``published_at`` to ``publishedAt``.

    Args:
        raw: Sort string from the LLM.

    Returns:
        Normalized sort value.
    """
    mapping = {
        "published_at": "publishedAt",
        "publishedat": "publishedAt",
        "published": "publishedAt",
        "relevancy": "relevancy",
        "relevant": "relevancy",
        "popularity": "popularity",
        "popular": "popularity",
    }
    return mapping.get(raw.strip().lower(), raw.strip())


# ---------------------------------------------------------------------------
# headlines
# ---------------------------------------------------------------------------

def headlines_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for top headlines.

    Args:
        **kwargs: Generic LLM params.  Uses ``api_key``, ``query``,
                  ``country``, ``category``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    api_key = kwargs.get("api_key", "")
    query = kwargs.get("query", "")
    country = kwargs.get("country", "us").strip().lower()
    category = kwargs.get("category", "")
    page_size = min(int(kwargs.get("limit", 20)), 100)

    params: dict[str, Any] = {
        "apiKey": api_key,
        "country": country,
        "pageSize": page_size,
    }
    if query:
        params["q"] = query
    if category:
        params["category"] = category

    return {"path": "/top-headlines", "params": params}


def headlines_parse(data: dict) -> list[dict[str, Any]]:
    """Parse NewsAPI top-headlines JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of article dicts with title, source, author, url, published_at,
        description.
    """
    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "author": a.get("author", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
            "description": a.get("description", ""),
        }
        for a in data.get("articles", [])
    ]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for searching all articles.

    Args:
        **kwargs: Generic LLM params.  Uses ``api_key``, ``query``,
                  ``sort_by``, ``language``, ``limit``.

    Returns:
        Request spec dict for http.fetch().
    """
    api_key = kwargs.get("api_key", "")
    query = kwargs.get("query", "")
    sort_by = _normalize_sort_by(kwargs.get("sort_by", "publishedAt"))
    language = _normalize_language(kwargs.get("language", "en"))
    page_size = min(int(kwargs.get("limit", 20)), 100)

    return {
        "path": "/everything",
        "params": {
            "apiKey": api_key,
            "q": query,
            "sortBy": sort_by,
            "language": language,
            "pageSize": page_size,
        },
    }


def search_parse(data: dict) -> list[dict[str, Any]]:
    """Parse NewsAPI everything-search JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of article dicts with title, source, author, url, published_at,
        description.
    """
    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "author": a.get("author", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
            "description": a.get("description", ""),
        }
        for a in data.get("articles", [])
    ]
