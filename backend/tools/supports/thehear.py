"""
The Hear multi-country headline aggregator request builders and response
parsers.

What it does:
    Defines request specs and response parsers for The Hear API.
    Fetches current headlines from 12-39 sources per country with
    AI-generated overviews.  Covers ideological diversity across 20
    countries.  No authentication required.

Entities in it:
    - BASE_URL: The Hear API root.
    - COUNTRY_ALIASES: Maps common country names and abbreviations to
      API-compatible lowercase slugs.
    - _normalize_country: Normalizes country names and codes to lowercase
      API-compatible slugs.
    - Request/parse pair for: country.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://www.thehear.org/api
"""

from typing import Any


BASE_URL = "https://www.thehear.org/api"

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

COUNTRY_ALIASES: dict[str, str] = {
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "united kingdom": "uk",
    "great britain": "uk",
    "gb": "uk",
    "germany": "de",
    "deutschland": "de",
    "france": "fr",
    "canada": "ca",
    "australia": "au",
    "japan": "jp",
    "india": "in",
    "brazil": "br",
    "mexico": "mx",
    "south korea": "kr",
    "korea": "kr",
    "italy": "it",
    "spain": "es",
    "netherlands": "nl",
    "sweden": "se",
    "switzerland": "ch",
    "norway": "no",
    "denmark": "dk",
    "finland": "fi",
    "poland": "pl",
    "turkey": "tr",
    "israel": "il",
    "south africa": "za",
    "nigeria": "ng",
    "argentina": "ar",
}


def _normalize_country(raw: str) -> str:
    """Normalize a country name or code to a lowercase API slug.

    Handles full country names (e.g. ``"United States"`` -> ``"us"``),
    common abbreviations, and raw codes.  Falls back to the lowercased,
    stripped input if no alias matches.

    Args:
        raw: Country name or code from the LLM.

    Returns:
        Lowercase country slug for the The Hear API path.
    """
    cleaned = raw.strip().lower()
    return COUNTRY_ALIASES.get(cleaned, cleaned)


# ---------------------------------------------------------------------------
# country
# ---------------------------------------------------------------------------

def country_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for headlines and overviews for a country.

    Args:
        **kwargs: Generic LLM params.  Uses ``country`` or ``query``
                  (normalized to a lowercase slug via _normalize_country).

    Returns:
        Request spec dict for http.fetch().
    """
    country = _normalize_country(kwargs.get("country", kwargs.get("query", "us")))
    return {"path": f"/country-view/{country}"}


def country_parse(data: dict) -> dict[str, Any]:
    """Parse The Hear country-view JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with headlines list and AI-generated overviews.
    """
    return data
