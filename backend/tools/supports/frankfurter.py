"""
Frankfurter foreign exchange rate connector.

Fetches daily FX rates from 84 central banks (201 currencies) back to 1948.
No authentication required. No rate limits.

API base: https://api.frankfurter.dev/v1
Docs: https://frankfurter.dev/
"""

from typing import Any

import httpx

BASE_URL = "https://api.frankfurter.dev/v1"


async def fetch_latest(
    base: str = "EUR",
    symbols: str = "",
) -> dict[str, Any]:
    """Fetch latest exchange rates.

    Args:
        base: Base currency code (e.g. "USD", "EUR").
        symbols: Comma-separated target currencies (empty = all).

    Returns:
        Dict with base, date, and rates mapping.
    """
    params: dict[str, Any] = {"base": base}
    if symbols:
        params["symbols"] = symbols
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{BASE_URL}/latest", params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_timeseries(
    base: str = "EUR",
    symbols: str = "USD",
    start_date: str = "",
    end_date: str = "",
) -> dict[str, Any]:
    """Fetch historical exchange rate time series.

    Args:
        base: Base currency code.
        symbols: Comma-separated target currencies.
        start_date: Start date YYYY-MM-DD (defaults to 30 days ago).
        end_date: End date YYYY-MM-DD (defaults to today).

    Returns:
        Dict with base, start_date, end_date, and rates keyed by date.
    """
    if not start_date or not end_date:
        from datetime import date, timedelta
        end_date = end_date or date.today().isoformat()
        start_date = start_date or (date.today() - timedelta(days=30)).isoformat()
    url = f"{BASE_URL}/{start_date}..{end_date}"
    params: dict[str, Any] = {"base": base}
    if symbols:
        params["symbols"] = symbols
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
