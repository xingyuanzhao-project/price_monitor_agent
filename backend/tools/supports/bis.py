"""
BIS (Bank for International Settlements) Statistics connector.

Fetches central bank policy rates, effective exchange rates, and credit data
from the BIS SDMX RESTful API. No authentication required.
Uses CSV format (the API does not support JSON).

API base: https://stats.bis.org/api/v1
Docs: https://stats.bis.org/api-doc/v1/
"""

import csv
import io
from typing import Any

import httpx

BASE_URL = "https://stats.bis.org/api/v1"


async def fetch_policy_rates(
    country: str = "US",
    frequency: str = "M",
    last_n: int = 24,
) -> dict[str, Any]:
    """Fetch central bank policy rates.

    Args:
        country: ISO 2-letter country code (US, GB, JP, CH, DE, etc.).
        frequency: M (monthly) or D (daily).
        last_n: Number of most recent observations.

    Returns:
        Dict with flow, key, and list of observations [{period, value}].
    """
    flow = "WS_CBPOL"
    key = f"{frequency}.{country}"
    return await _fetch_bis_data(flow, key, last_n)


async def fetch_exchange_rates(
    country: str = "US",
    rate_type: str = "N",
    basis: str = "B",
    last_n: int = 24,
) -> dict[str, Any]:
    """Fetch BIS effective exchange rates (REER/NEER).

    Args:
        country: ISO 2-letter country code.
        rate_type: N (nominal) or R (real).
        basis: B (broad) or N (narrow).
        last_n: Number of most recent observations.

    Returns:
        Dict with flow, key, and list of observations [{period, value}].
    """
    flow = "WS_EER"
    key = f"M.{rate_type}.{basis}.{country}"
    return await _fetch_bis_data(flow, key, last_n)


async def _fetch_bis_data(flow: str, key: str, last_n: int) -> dict[str, Any]:
    url = f"{BASE_URL}/data/{flow}/{key}/all"
    params = {"lastNObservations": last_n, "detail": "dataonly", "format": "csv"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return _parse_csv(response.text, flow, key)


def _parse_csv(text: str, flow: str, key: str) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(text))
    observations = []
    for row in reader:
        period = row.get("TIME_PERIOD", "")
        value_string = row.get("OBS_VALUE", "")
        try:
            value = float(value_string)
        except (ValueError, TypeError):
            value = value_string
        observations.append({"period": period, "value": value})
    return {"flow": flow, "key": key, "observations": observations}
