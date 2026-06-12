"""
Frankfurter foreign exchange rate request builders and response parsers.

What it does:
    Defines request specs and response parsers for the Frankfurter FX API.
    Covers latest rates and historical time-series for 201 currencies from
    84 central banks back to 1948.  No authentication required, no rate limits.

Entities in it:
    - BASE_URL: Frankfurter API v1 root.
    - _normalize_currency: Uppercases 3-letter currency codes.
    - Request/parse pairs for: latest, timeseries.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://frankfurter.dev/
"""

from typing import Any


BASE_URL = "https://api.frankfurter.dev/v1"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_currency(raw: str) -> str:
    """Uppercase and strip a 3-letter currency code.

    Args:
        raw: Currency code from the LLM (e.g. "eur", " usd ").

    Returns:
        Uppercased 3-letter currency code.
    """
    return raw.strip().upper()


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------

def latest_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for latest exchange rates.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (base currency),
                  ``symbols`` (comma-separated target currencies).

    Returns:
        Request spec dict for http.fetch().
    """
    base = _normalize_currency(kwargs.get("symbol", "") or kwargs.get("base", "EUR"))
    symbols_raw = kwargs.get("symbols", "")
    params: dict[str, Any] = {"base": base}
    if symbols_raw:
        params["symbols"] = _normalize_currency(symbols_raw)
    return {
        "path": "/latest",
        "params": params,
    }


def latest_parse(data: dict) -> dict[str, Any]:
    """Parse Frankfurter latest rates JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with base, date, and rates mapping.
    """
    return data


# ---------------------------------------------------------------------------
# timeseries
# ---------------------------------------------------------------------------

def timeseries_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for historical exchange rate time series.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (base currency),
                  ``symbols`` (target currencies), ``start_date``,
                  ``end_date`` (YYYY-MM-DD).

    Returns:
        Request spec dict for http.fetch().
    """
    base = _normalize_currency(kwargs.get("symbol", "") or kwargs.get("base", "EUR"))
    symbols_raw = kwargs.get("symbols", "USD")
    start_date = kwargs.get("start_date", "")
    end_date = kwargs.get("end_date", "")

    if not start_date or not end_date:
        from datetime import date, timedelta
        end_date = end_date or date.today().isoformat()
        start_date = start_date or (date.today() - timedelta(days=30)).isoformat()

    params: dict[str, Any] = {"base": base}
    if symbols_raw:
        params["symbols"] = _normalize_currency(symbols_raw)

    return {
        "path": f"/{start_date}..{end_date}",
        "params": params,
    }


def timeseries_parse(data: dict) -> dict[str, Any]:
    """Parse Frankfurter timeseries JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with base, start_date, end_date, and rates keyed by date.
    """
    return data
