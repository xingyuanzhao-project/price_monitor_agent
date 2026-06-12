"""
BIS (Bank for International Settlements) Statistics request builders and
response parsers.

What it does:
    Defines request specs and response parsers for the BIS SDMX RESTful API.
    Covers central bank policy rates and effective exchange rates (REER/NEER).
    Returns CSV format (the API does not support JSON).
    No authentication required.

Entities in it:
    - BASE_URL: BIS Statistics API v1 root.
    - _normalize_country: Uppercases 2-letter ISO country codes.
    - _normalize_frequency: Uppercases single-letter frequency (M/D).
    - _parse_csv: Shared CSV text parser for BIS responses.
    - Request/parse pairs for: policy_rates, exchange_rates.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw CSV text to the parse function.

API docs: https://stats.bis.org/api-doc/v1/
"""

import csv
import io
from typing import Any


BASE_URL = "https://stats.bis.org/api/v1"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_country(raw: str) -> str:
    """Uppercase and strip a 2-letter ISO country code.

    Args:
        raw: Country code from the LLM (e.g. "us", " GB ").

    Returns:
        Uppercased 2-letter country code.
    """
    return raw.strip().upper()


def _normalize_frequency(raw: str) -> str:
    """Uppercase a single-letter frequency code.

    Accepts M (monthly) or D (daily).

    Args:
        raw: Frequency from the LLM (e.g. "m", "D").

    Returns:
        Single uppercase letter.
    """
    return raw.strip().upper()[:1] or "M"


# ---------------------------------------------------------------------------
# Shared CSV parser
# ---------------------------------------------------------------------------

def _parse_csv(text: str, flow: str) -> dict[str, Any]:
    """Parse BIS SDMX CSV response text into structured observations.

    Args:
        text: Raw CSV text from the API.
        flow: BIS dataflow identifier (e.g. "WS_CBPOL").

    Returns:
        Dict with flow and list of observations [{period, value}].
    """
    reader = csv.DictReader(io.StringIO(text))
    observations = []
    for row in reader:
        period = row.get("TIME_PERIOD", "")
        value_string = row.get("OBS_VALUE", "")
        try:
            value: Any = float(value_string)
        except (ValueError, TypeError):
            value = value_string
        observations.append({"period": period, "value": value})
    return {"flow": flow, "observations": observations}


# ---------------------------------------------------------------------------
# policy_rates
# ---------------------------------------------------------------------------

def policy_rates_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for central bank policy rates.

    Args:
        **kwargs: Generic LLM params.  Uses ``country`` (2-letter ISO),
                  ``frequency`` (M or D), ``limit`` (lastNObservations).

    Returns:
        Request spec dict for http.fetch().
    """
    country = _normalize_country(kwargs.get("country", "US"))
    frequency = _normalize_frequency(kwargs.get("frequency", "M"))
    limit = int(kwargs.get("limit", 24))
    key = f"{frequency}.{country}"
    return {
        "path": f"/data/WS_CBPOL/{key}/all",
        "params": {
            "lastNObservations": limit,
            "detail": "dataonly",
            "format": "csv",
        },
        "response_format": "text",
        "timeout": 20.0,
    }


def policy_rates_parse(data: str) -> dict[str, Any]:
    """Parse BIS policy rates CSV response.

    Args:
        data: Raw CSV text from the API.

    Returns:
        Dict with flow and list of observations [{period, value}].
    """
    return _parse_csv(data, "WS_CBPOL")


# ---------------------------------------------------------------------------
# exchange_rates
# ---------------------------------------------------------------------------

def exchange_rates_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for BIS effective exchange rates (REER/NEER).

    Args:
        **kwargs: Generic LLM params.  Uses ``country`` (2-letter ISO),
                  ``indicator`` (N for nominal, R for real),
                  ``basis`` (B for broad, N for narrow),
                  ``limit`` (lastNObservations).

    Returns:
        Request spec dict for http.fetch().
    """
    country = _normalize_country(kwargs.get("country", "US"))
    rate_type = kwargs.get("indicator", "N").strip().upper()[:1]
    basis = kwargs.get("basis", "B").strip().upper()[:1]
    limit = int(kwargs.get("limit", 24))
    key = f"M.{rate_type}.{basis}.{country}"
    return {
        "path": f"/data/WS_EER/{key}/all",
        "params": {
            "lastNObservations": limit,
            "detail": "dataonly",
            "format": "csv",
        },
        "response_format": "text",
        "timeout": 20.0,
    }


def exchange_rates_parse(data: str) -> dict[str, Any]:
    """Parse BIS effective exchange rate CSV response.

    Args:
        data: Raw CSV text from the API.

    Returns:
        Dict with flow and list of observations [{period, value}].
    """
    return _parse_csv(data, "WS_EER")
