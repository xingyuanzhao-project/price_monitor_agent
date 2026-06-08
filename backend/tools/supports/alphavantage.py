"""
Alpha Vantage API support.

Fetches stock, forex, and crypto data from Alpha Vantage.
Requires a free API key (register at alphavantage.co).

API base: https://www.alphavantage.co/query
Docs: https://www.alphavantage.co/documentation/
"""

from typing import Any

import httpx

BASE_URL = "https://www.alphavantage.co/query"


async def fetch_quote(symbol: str, api_key: str) -> dict[str, Any]:
    """Fetch global quote for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. "AAPL", "MSFT").
        api_key: Alpha Vantage API key.

    Returns:
        Normalized dict with symbol, price, change, change_pct, volume.
    """
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json().get("Global Quote", {})
        return {
            "symbol": data.get("01. symbol", symbol),
            "price": data.get("05. price"),
            "change": data.get("09. change"),
            "change_pct": data.get("10. change percent"),
            "volume": data.get("06. volume"),
        }


async def fetch_daily(
    symbol: str,
    api_key: str,
    outputsize: str = "compact",
) -> list[dict[str, Any]]:
    """Fetch daily time series (OHLCV).

    Args:
        symbol: Stock ticker.
        api_key: Alpha Vantage API key.
        outputsize: "compact" (100 days) or "full" (20+ years).

    Returns:
        List of daily bar dicts with date, o, h, l, c, vol.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        ts = resp.json().get("Time Series (Daily)", {})
        bars = []
        for date, values in ts.items():
            bars.append({
                "date": date,
                "o": values["1. open"],
                "h": values["2. high"],
                "l": values["3. low"],
                "c": values["4. close"],
                "vol": values["5. volume"],
            })
        return bars


async def fetch_crypto_exchange_rate(
    from_currency: str,
    to_currency: str,
    api_key: str,
) -> dict[str, Any]:
    """Fetch real-time crypto exchange rate.

    Args:
        from_currency: Crypto code (e.g. "BTC").
        to_currency: Target currency (e.g. "USD").
        api_key: Alpha Vantage API key.

    Returns:
        Dict with from, to, rate, last_refreshed.
    """
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency,
        "to_currency": to_currency,
        "apikey": api_key,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json().get("Realtime Currency Exchange Rate", {})
        return {
            "from": data.get("1. From_Currency Code"),
            "to": data.get("3. To_Currency Code"),
            "rate": data.get("5. Exchange Rate"),
            "last_refreshed": data.get("6. Last Refreshed"),
        }
