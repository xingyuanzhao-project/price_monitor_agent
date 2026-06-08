"""
World Bank Open Data connector.

Fetches development indicators for 200+ countries from the World Bank API.
No authentication required.

API base: https://api.worldbank.org/v2
Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581
"""

from typing import Any

import httpx

BASE_URL = "https://api.worldbank.org/v2"


async def fetch_indicator(
    indicator: str,
    country: str = "US",
    date_range: str = "2015:2025",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch a development indicator for a country.

    Args:
        indicator: World Bank indicator code (e.g. "NY.GDP.MKTP.CD" for GDP,
                   "FP.CPI.TOTL.ZG" for inflation, "SL.UEM.TOTL.ZS" for unemployment).
        country: ISO 2-letter country code or "all".
        date_range: Year range "YYYY:YYYY".
        limit: Max records per page.

    Returns:
        List of observation dicts with country, date, value, indicator.
    """
    url = f"{BASE_URL}/country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "date": date_range,
        "per_page": min(limit, 500),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2:
            return []
        records = data[1] or []
        return [
            {
                "country": r.get("country", {}).get("value", ""),
                "country_code": r.get("countryiso3code", ""),
                "date": r.get("date", ""),
                "value": r.get("value"),
                "indicator": r.get("indicator", {}).get("id", ""),
                "indicator_name": r.get("indicator", {}).get("value", ""),
            }
            for r in records
        ]


async def search_indicators(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search for World Bank indicators by keyword.

    Args:
        query: Search term (e.g. "GDP", "inflation", "trade").
        limit: Max results.

    Returns:
        List of indicator dicts with id, name, source.
    """
    url = f"{BASE_URL}/indicator"
    params = {
        "format": "json",
        "per_page": 1000,
        "source": "2",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2:
            return []
        indicators = data[1] or []
        q = query.lower()
        matched = [
            {
                "id": ind.get("id", ""),
                "name": ind.get("name", ""),
                "source": ind.get("source", {}).get("value", ""),
            }
            for ind in indicators
            if q in ind.get("name", "").lower() or q in ind.get("id", "").lower()
        ]
        return matched[:limit]
