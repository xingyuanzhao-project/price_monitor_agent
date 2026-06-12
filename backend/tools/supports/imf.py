"""
IMF DataMapper request builders and response parsers.

What it does:
    Defines request specs and response parsers for the IMF public JSON REST
    API.  Covers macroeconomic indicators (GDP growth, inflation, unemployment,
    current account, government debt) for approximately 190 countries.
    No authentication required.

Entities in it:
    - BASE_URL: IMF DataMapper API v1 root.
    - _normalize_country_codes: Uppercases comma-separated ISO 3-letter codes.
    - Request/parse pairs for: indicator, list.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://datahelp.imf.org/knowledgebase/articles/667681
"""

from typing import Any


BASE_URL = "https://www.imf.org/external/datamapper/api/v1"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_country_codes(raw: str) -> list[str]:
    """Uppercase and split comma-separated ISO 3-letter country codes.

    Args:
        raw: Comma-separated country codes (e.g. "usa,gbr,chn").

    Returns:
        List of uppercased country code strings.
    """
    return [c.strip().upper() for c in raw.split(",") if c.strip()]


def _normalize_indicator(raw: str) -> str:
    """Uppercase an IMF indicator code.

    Args:
        raw: Indicator code from the LLM (e.g. "ngdp_rpch").

    Returns:
        Uppercased indicator code.
    """
    return raw.strip().upper()


# ---------------------------------------------------------------------------
# indicator
# ---------------------------------------------------------------------------

def indicator_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for an IMF indicator across countries.

    Args:
        **kwargs: Generic LLM params.  Uses ``indicator``, ``country``
                  (comma-separated ISO3 codes), ``periods`` (comma-separated
                  years).

    Returns:
        Request spec dict for http.fetch().
    """
    indicator = _normalize_indicator(kwargs.get("indicator", "NGDP_RPCH"))
    countries_raw = kwargs.get("country", "USA")
    periods = kwargs.get("periods", "")

    country_list = _normalize_country_codes(countries_raw)
    country_path = "/".join(country_list) if country_list else ""

    path = f"/{indicator}"
    if country_path:
        path += f"/{country_path}"

    params: dict[str, Any] = {}
    if periods:
        params["periods"] = periods

    return {
        "path": path,
        "params": params if params else None,
        "timeout": 15.0,
    }


def indicator_parse(data: dict) -> dict[str, Any]:
    """Parse IMF indicator JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with indicator code and data keyed by country then year.
    """
    indicator_keys = [k for k in data.get("values", {}).keys()]
    indicator = indicator_keys[0] if indicator_keys else ""
    values = data.get("values", {}).get(indicator, {})
    return {
        "indicator": indicator,
        "data": values,
    }


# ---------------------------------------------------------------------------
# list (all available indicators)
# ---------------------------------------------------------------------------

def list_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for listing all IMF DataMapper indicators.

    Args:
        **kwargs: Generic LLM params.  None required.

    Returns:
        Request spec dict for http.fetch().
    """
    return {
        "path": "/indicators",
        "timeout": 15.0,
    }


def list_parse(data: dict) -> list[dict[str, Any]]:
    """Parse IMF indicator listing JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of dicts with indicator code, label, and description.
    """
    indicators = data.get("indicators", {})
    return [
        {
            "code": code,
            "label": meta.get("label", ""),
            "description": meta.get("description", ""),
        }
        for code, meta in indicators.items()
    ]
