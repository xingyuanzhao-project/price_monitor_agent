"""
FRED (Federal Reserve Economic Data) public API support.

Fetches US macroeconomic indicators from the FRED API.
Requires a free API key (register at fredaccount.stlouisfed.org).

API base: https://api.stlouisfed.org/fred/
Docs: https://fred.stlouisfed.org/docs/api/fred/
"""

from typing import Any

import httpx

BASE_URL = "https://api.stlouisfed.org/fred"


async def fetch_series_observations(
    series_id: str,
    api_key: str,
    limit: int = 100,
    sort_order: str = "desc",
) -> list[dict[str, Any]]:
    """Fetch observations for a FRED series.

    Args:
        series_id: FRED series ID (e.g. "GDP", "CPIAUCSL", "UNRATE", "DFF").
        api_key: FRED API key (free registration at fredaccount.stlouisfed.org).
        limit: Number of observations to return.
        sort_order: "asc" or "desc".

    Returns:
        List of observation dicts with date, value.
    """
    url = f"{BASE_URL}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit,
        "sort_order": sort_order,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return [
            {"date": obs["date"], "value": obs["value"]}
            for obs in data.get("observations", [])
            if obs.get("value") != "."
        ]


async def fetch_series_info(series_id: str, api_key: str) -> dict[str, Any]:
    """Fetch metadata for a FRED series.

    Args:
        series_id: FRED series ID.
        api_key: FRED API key.

    Returns:
        Dict with id, title, frequency, units, seasonal_adjustment.
    """
    url = f"{BASE_URL}/series"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        series = data.get("seriess", [{}])[0]
        return {
            "id": series.get("id"),
            "title": series.get("title"),
            "frequency": series.get("frequency"),
            "units": series.get("units"),
            "seasonal_adjustment": series.get("seasonal_adjustment"),
        }


async def search_series(
    query: str,
    api_key: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search for FRED series by keyword.

    Args:
        query: Search text.
        api_key: FRED API key.
        limit: Max results.

    Returns:
        List of series dicts with id, title, frequency, popularity.
    """
    url = f"{BASE_URL}/series/search"
    params = {"search_text": query, "api_key": api_key, "file_type": "json", "limit": limit}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "id": s["id"],
                "title": s["title"],
                "frequency": s.get("frequency"),
                "popularity": s.get("popularity"),
            }
            for s in data.get("seriess", [])
        ]
