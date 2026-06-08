"""
Nasdaq Data Link (formerly Quandl) API support.

Fetches economic, financial, and alternative datasets from Nasdaq Data Link.
Requires an API key.

API base: https://data.nasdaq.com/api/v3/
Docs: https://docs.data.nasdaq.com/
"""

from typing import Any

import httpx

BASE_URL = "https://data.nasdaq.com/api/v3"


async def fetch_dataset(
    database_code: str,
    dataset_code: str,
    api_key: str,
    limit: int = 100,
    order: str = "desc",
) -> list[dict[str, Any]]:
    """Fetch a time-series dataset.

    Args:
        database_code: Database code (e.g. "FRED", "WIKI", "EOD").
        dataset_code: Dataset code (e.g. "GDP", "AAPL").
        api_key: Nasdaq Data Link API key.
        limit: Number of rows.
        order: "asc" or "desc".

    Returns:
        List of row dicts with column names from the dataset.
    """
    url = f"{BASE_URL}/datasets/{database_code}/{dataset_code}/data.json"
    params = {"api_key": api_key, "limit": limit, "order": order}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("dataset_data", {})
        columns = data.get("column_names", [])
        rows = []
        for row in data.get("data", []):
            rows.append(dict(zip(columns, row)))
        return rows


async def fetch_dataset_metadata(
    database_code: str,
    dataset_code: str,
    api_key: str,
) -> dict[str, Any]:
    """Fetch metadata for a dataset.

    Args:
        database_code: Database code.
        dataset_code: Dataset code.
        api_key: Nasdaq Data Link API key.

    Returns:
        Dict with name, description, frequency, column_names.
    """
    url = f"{BASE_URL}/datasets/{database_code}/{dataset_code}/metadata.json"
    params = {"api_key": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("dataset", {})
        return {
            "name": data.get("name"),
            "description": data.get("description", "")[:500],
            "frequency": data.get("frequency"),
            "column_names": data.get("column_names", []),
            "newest_available_date": data.get("newest_available_date"),
            "oldest_available_date": data.get("oldest_available_date"),
        }
