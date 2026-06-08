"""
USA Spending connector.

Fetches federal spending data from the US government's open data API.
No authentication required.

API base: https://api.usaspending.gov/api/v2
Docs: https://api.usaspending.gov/docs/endpoints
"""

from typing import Any

import httpx

BASE_URL = "https://api.usaspending.gov/api/v2"


async def fetch_spending_by_agency(
    fiscal_year: int = 2025,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch total federal spending by agency for a fiscal year.

    Args:
        fiscal_year: US fiscal year (e.g. 2025).
        limit: Max agencies.

    Returns:
        List of agency spending dicts with name, budget, obligations.
    """
    url = f"{BASE_URL}/spending/"
    payload = {
        "type": "agency",
        "filters": {"fy": str(fiscal_year), "quarter": "4"},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        return results[:limit]


async def fetch_spending_over_time(
    group: str = "fiscal_year",
    time_period_start: str = "2020-10-01",
    time_period_end: str = "2025-09-30",
) -> list[dict[str, Any]]:
    """Fetch federal spending aggregated over time.

    Args:
        group: Grouping -- "fiscal_year", "quarter", or "month".
        time_period_start: Start date YYYY-MM-DD.
        time_period_end: End date YYYY-MM-DD.

    Returns:
        List of time-period dicts with aggregated_amount.
    """
    url = f"{BASE_URL}/search/spending_over_time/"
    payload = {
        "group": group,
        "filters": {
            "time_period": [
                {"start_date": time_period_start, "end_date": time_period_end}
            ],
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
