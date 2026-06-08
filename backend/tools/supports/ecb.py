"""
ECB (European Central Bank) Statistical Data Warehouse support.

Fetches euro area economic and financial statistics from the ECB SDMX API.
No authentication required.

API base: https://data-api.ecb.europa.eu/service/data/
Docs: https://data.ecb.europa.eu/help/api/overview
"""

from typing import Any

import httpx

BASE_URL = "https://data-api.ecb.europa.eu/service/data"


async def fetch_exchange_rates(
    currency: str = "USD",
    frequency: str = "D",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch EUR exchange rate series.

    Args:
        currency: Target currency code (e.g. "USD", "GBP", "JPY").
        frequency: D (daily), M (monthly), A (annual).
        limit: Number of recent observations.

    Returns:
        List of dicts with date, value (EUR/target rate).
    """
    flow_ref = "EXR"
    key = f"{frequency}.{currency}.EUR.SP00.A"
    url = f"{BASE_URL}/{flow_ref}/{key}"
    params = {"format": "jsondata", "lastNObservations": limit}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        observations = []
        datasets = data.get("dataSets", [{}])
        if datasets:
            series_data = datasets[0].get("series", {})
            for _key, series in series_data.items():
                obs = series.get("observations", {})
                for idx, values in obs.items():
                    observations.append({"index": int(idx), "value": values[0]})
        return observations


async def fetch_interest_rates(
    rate_type: str = "MRR_FR",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch ECB key interest rates.

    Args:
        rate_type: Rate identifier — MRR_FR (main refinancing, fixed rate), DFR (deposit facility), MLFR (marginal lending).
        limit: Number of recent observations.

    Returns:
        List of dicts with index, value.
    """
    flow_ref = "FM"
    key = f"D.U2.EUR.4F.KR.{rate_type}.LEV"
    url = f"{BASE_URL}/{flow_ref}/{key}"
    params = {"format": "jsondata", "lastNObservations": limit}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        observations = []
        datasets = data.get("dataSets", [{}])
        if datasets:
            series_data = datasets[0].get("series", {})
            for _key, series in series_data.items():
                obs = series.get("observations", {})
                for idx, values in obs.items():
                    observations.append({"index": int(idx), "value": values[0]})
        return observations
