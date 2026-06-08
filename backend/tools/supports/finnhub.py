"""
Finnhub API support.

Fetches stock market data, company profiles, earnings, and news from Finnhub.
Requires a free API key.

API base: https://finnhub.io/api/v1/
Docs: https://finnhub.io/docs/api
"""

from typing import Any

import httpx

BASE_URL = "https://finnhub.io/api/v1"


async def fetch_quote(symbol: str, api_key: str) -> dict[str, Any]:
    """Fetch real-time quote.

    Args:
        symbol: Stock ticker (e.g. "AAPL").
        api_key: Finnhub API key.

    Returns:
        Dict with current, high, low, open, prev_close, change, change_pct.
    """
    url = f"{BASE_URL}/quote"
    params = {"symbol": symbol, "token": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return {
            "current": data.get("c"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "prev_close": data.get("pc"),
            "change": data.get("d"),
            "change_pct": data.get("dp"),
        }


async def fetch_company_news(
    symbol: str,
    api_key: str,
    from_date: str,
    to_date: str,
) -> list[dict[str, Any]]:
    """Fetch company news articles.

    Args:
        symbol: Stock ticker.
        api_key: Finnhub API key.
        from_date: Start date (YYYY-MM-DD), required by the API.
        to_date: End date (YYYY-MM-DD), required by the API.

    Returns:
        List of article dicts with headline, source, url, datetime, summary.
    """
    url = f"{BASE_URL}/company-news"
    params = {"symbol": symbol, "from": from_date, "to": to_date, "token": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        articles = []
        for a in resp.json():
            articles.append({
                "headline": a.get("headline", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "datetime": a.get("datetime"),
                "summary": a.get("summary", "")[:500],
            })
        return articles


async def fetch_earnings(symbol: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch earnings surprises.

    Args:
        symbol: Stock ticker.
        api_key: Finnhub API key.

    Returns:
        List of earnings dicts with period, actual, estimate, surprise, surprise_pct.
    """
    url = f"{BASE_URL}/stock/earnings"
    params = {"symbol": symbol, "token": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        earnings = []
        for e in resp.json():
            earnings.append({
                "period": e.get("period"),
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "surprise": e.get("surprise"),
                "surprise_pct": e.get("surprisePercent"),
            })
        return earnings
