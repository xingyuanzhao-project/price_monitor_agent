"""
USA Spending request builders and response parsers.

What it does:
    Defines request specs and response parsers for the USAspending.gov API v2.
    Fetches federal spending data by agency and over time.  Uses POST requests.
    No authentication required.

Entities in it:
    - BASE_URL: USAspending API v2 root.
    - VALID_GROUPS: Set of accepted spending aggregation group values.
    - _normalize_fiscal_year: Coerces fiscal year values to valid integers.
    - _normalize_group: Validates spending aggregation group values against
      allowed options.
    - _normalize_date: Normalizes date strings to YYYY-MM-DD format.
    - Request/parse pairs for: by_agency, over_time.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://api.usaspending.gov/docs/endpoints
"""

from typing import Any


BASE_URL = "https://api.usaspending.gov/api/v2"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

VALID_GROUPS = {"fiscal_year", "quarter", "month"}


def _normalize_fiscal_year(raw: Any, default: int = 2025) -> int:
    """Coerce a fiscal year to a valid integer.

    Args:
        raw: Fiscal year from the LLM (int, float, or str).
        default: Fallback when raw is falsy or unparseable.

    Returns:
        Integer fiscal year.
    """
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return default


def _normalize_group(raw: str) -> str:
    """Validate a spending aggregation group value.

    Accepted values are ``fiscal_year``, ``quarter``, and ``month``.
    Handles common separator variants (spaces, hyphens) and falls back
    to ``"fiscal_year"`` for unrecognized input.

    Args:
        raw: Group value from the LLM.

    Returns:
        Validated group string.
    """
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in VALID_GROUPS:
        return cleaned
    return "fiscal_year"


def _normalize_date(raw: str) -> str:
    """Normalize a date string to YYYY-MM-DD format.

    Handles common LLM variants: YYYY/MM/DD, MM-DD-YYYY,
    dates with trailing whitespace.

    Args:
        raw: Date string from the LLM.

    Returns:
        Date in YYYY-MM-DD format.
    """
    s = raw.strip()
    s = s.replace("/", "-")
    parts = s.split("-")
    if len(parts) == 3 and len(parts[0]) <= 2 and len(parts[2]) == 4:
        s = f"{parts[2]}-{parts[0]}-{parts[1]}"
    return s


# ---------------------------------------------------------------------------
# by_agency
# ---------------------------------------------------------------------------

def by_agency_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for total federal spending by agency.

    Args:
        **kwargs: Generic LLM params.  Uses ``fiscal_year`` (integer),
                  ``limit`` (max agencies to return).

    Returns:
        Request spec dict for http.fetch().
    """
    fiscal_year = _normalize_fiscal_year(kwargs.get("fiscal_year", 2025))
    limit = int(kwargs.get("limit", 20))
    return {
        "path": "/spending/",
        "method": "POST",
        "body": {
            "type": "agency",
            "filters": {"fy": str(fiscal_year), "quarter": "4"},
        },
        "timeout": 20.0,
    }


def by_agency_parse(data: dict) -> list[dict[str, Any]]:
    """Parse USAspending agency spending JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of agency spending dicts.
    """
    return data.get("results", [])


# ---------------------------------------------------------------------------
# over_time
# ---------------------------------------------------------------------------

def over_time_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for federal spending aggregated over time.

    Args:
        **kwargs: Generic LLM params.  Uses ``group`` (fiscal_year, quarter,
                  or month), ``start_date`` (YYYY-MM-DD),
                  ``end_date`` (YYYY-MM-DD).

    Returns:
        Request spec dict for http.fetch().
    """
    group = _normalize_group(kwargs.get("group", "fiscal_year"))
    start_date = _normalize_date(kwargs.get("start_date", "2020-10-01"))
    end_date = _normalize_date(kwargs.get("end_date", "2025-09-30"))
    return {
        "path": "/search/spending_over_time/",
        "method": "POST",
        "body": {
            "group": group,
            "filters": {
                "time_period": [
                    {"start_date": start_date, "end_date": end_date}
                ],
            },
        },
        "timeout": 30.0,
    }


def over_time_parse(data: dict) -> list[dict[str, Any]]:
    """Parse USAspending spending-over-time JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of time-period dicts with aggregated_amount.
    """
    return data.get("results", [])
