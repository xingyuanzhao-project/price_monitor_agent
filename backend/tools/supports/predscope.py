"""
PredScope API request builders and response parsers.

What it does:
    Defines request specs and response parsers for PredScope's free REST API.
    Covers top-100 active prediction markets and recently resolved markets.
    No authentication required. Rate limit: 100 req/hour.

Entities in it:
    - BASE_URL: PredScope API root.
    - _normalize_limit: Coerces and clamps result-count limits to valid
      integer bounds.
    - Request/parse pairs for: markets, resolved.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://predscope.com/api
"""

from typing import Any


BASE_URL = "https://predscope.com/api"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_limit(raw: Any, default: int = 100, lower: int = 1, upper: int = 100) -> int:
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
# markets
# ---------------------------------------------------------------------------

def markets_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for top-100 active prediction markets.

    Args:
        **kwargs: Generic LLM params.  Uses ``limit`` (applied client-side
                  after parsing).

    Returns:
        Request spec dict for http.fetch().
    """
    limit = _normalize_limit(kwargs.get("limit", 100))
    return {"path": "/markets.json", "params": {}, "limit": limit, "timeout": 15.0}


def markets_parse(data: dict) -> dict[str, Any]:
    """Parse PredScope active markets JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with meta and markets list as returned by the API.
    """
    return data


# ---------------------------------------------------------------------------
# resolved
# ---------------------------------------------------------------------------

def resolved_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for recently resolved prediction markets.

    Args:
        **kwargs: Generic LLM params.  Uses ``limit`` (applied client-side
                  after parsing).

    Returns:
        Request spec dict for http.fetch().
    """
    limit = _normalize_limit(kwargs.get("limit", 100))
    return {"path": "/resolved.json", "params": {}, "limit": limit, "timeout": 15.0}


def resolved_parse(data: dict) -> dict[str, Any]:
    """Parse PredScope resolved markets JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with resolved market data as returned by the API.
    """
    return data
