"""
ECB (European Central Bank) Statistical Data Warehouse request builders and
response parsers.

What it does:
    Defines request specs and response parsers for the ECB SDMX JSON API.
    Covers euro area exchange rates and key interest rates (MRR, DFR, MLFR).
    No authentication required.

Entities in it:
    - BASE_URL: ECB Statistical Data Warehouse API root.
    - _normalize_currency: Uppercases and strips whitespace from currency codes.
    - Request/parse pairs for: exchange_rates, interest_rates.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://data.ecb.europa.eu/help/api/overview
"""

from typing import Any


BASE_URL = "https://data-api.ecb.europa.eu/service/data"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_currency(raw: str) -> str:
    """Uppercase and strip a currency code.

    Args:
        raw: Currency string from the LLM (e.g. "usd", " GBP ").

    Returns:
        Uppercased, stripped currency code.
    """
    return raw.strip().upper()


def _normalize_frequency(raw: str) -> str:
    """Uppercase and validate a frequency code.

    Accepts D (daily), M (monthly), A (annual).

    Args:
        raw: Frequency string from the LLM.

    Returns:
        Single uppercase letter frequency code.
    """
    return raw.strip().upper()[:1] or "D"


# ---------------------------------------------------------------------------
# exchange_rates
# ---------------------------------------------------------------------------

def exchange_rates_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for EUR exchange rate series.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (target currency),
                  ``frequency`` (D/M/A), ``limit`` (lastNObservations).

    Returns:
        Request spec dict for http.fetch().
    """
    currency = _normalize_currency(kwargs.get("symbol", "USD"))
    frequency = _normalize_frequency(kwargs.get("frequency", "D"))
    limit = int(kwargs.get("limit", 100))
    series_key = f"{frequency}.{currency}.EUR.SP00.A"
    return {
        "path": f"/EXR/{series_key}",
        "params": {"format": "jsondata", "lastNObservations": limit},
    }


def exchange_rates_parse(data: dict) -> list[dict[str, Any]]:
    """Parse ECB exchange rate JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of observation dicts with index and value.
    """
    observations: list[dict[str, Any]] = []
    datasets = data.get("dataSets", [{}])
    if datasets:
        series_data = datasets[0].get("series", {})
        for _key, series in series_data.items():
            obs = series.get("observations", {})
            for idx, values in obs.items():
                observations.append({"index": int(idx), "value": values[0]})
    return observations


# ---------------------------------------------------------------------------
# interest_rates
# ---------------------------------------------------------------------------

def interest_rates_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for ECB key interest rates.

    Args:
        **kwargs: Generic LLM params.  Uses ``indicator`` (rate type:
                  MRR_FR, DFR, MLFR), ``limit`` (lastNObservations).

    Returns:
        Request spec dict for http.fetch().
    """
    rate_type = kwargs.get("indicator", "MRR_FR").strip().upper()
    limit = int(kwargs.get("limit", 50))
    series_key = f"D.U2.EUR.4F.KR.{rate_type}.LEV"
    return {
        "path": f"/FM/{series_key}",
        "params": {"format": "jsondata", "lastNObservations": limit},
    }


def interest_rates_parse(data: dict) -> list[dict[str, Any]]:
    """Parse ECB interest rate JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of observation dicts with index and value.
    """
    observations: list[dict[str, Any]] = []
    datasets = data.get("dataSets", [{}])
    if datasets:
        series_data = datasets[0].get("series", {})
        for _key, series in series_data.items():
            obs = series.get("observations", {})
            for idx, values in obs.items():
                observations.append({"index": int(idx), "value": values[0]})
    return observations
