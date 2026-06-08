"""
Yahoo Finance public data connector.

Fetches equity, ETF, and index data from Yahoo Finance's unofficial chart API.
No authentication required. Data is 15-min delayed during US market hours.

Endpoint: https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
"""

from typing import Any

import httpx

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {"User-Agent": "price_monitor_agent/1.0"}


async def fetch_quote(symbol: str) -> dict[str, Any]:
    """Fetch current quote for a stock, ETF, or index.

    Args:
        symbol: Yahoo Finance ticker (e.g. "AAPL", "^GSPC", "EURUSD=X").

    Returns:
        Dict with symbol, price, change, volume, market state.
    """
    url = f"{BASE_URL}/{symbol}"
    params = {"interval": "1d", "range": "1d"}
    async with httpx.AsyncClient(headers=HEADERS, timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        return {
            "symbol": meta["symbol"],
            "price": meta.get("regularMarketPrice"),
            "previous_close": meta.get("previousClose"),
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName"),
            "market_state": meta.get("marketState"),
        }


async def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    range_period: str = "1mo",
) -> list[dict[str, Any]]:
    """Fetch OHLCV candlestick history for a stock, ETF, or index.

    Args:
        symbol: Yahoo Finance ticker (e.g. "AAPL", "^GSPC").
        interval: Bar interval ("1m","5m","15m","1h","1d","1wk","1mo").
        range_period: Lookback period ("1d","5d","1mo","3mo","6mo","1y","5y","max").

    Returns:
        List of candle dicts with ts, o, h, l, c, vol.
    """
    url = f"{BASE_URL}/{symbol}"
    params = {"interval": interval, "range": range_period}
    async with httpx.AsyncClient(headers=HEADERS, timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        candles = []
        for i, ts in enumerate(timestamps):
            candles.append({
                "ts": ts,
                "o": quotes.get("open", [None])[i],
                "h": quotes.get("high", [None])[i],
                "l": quotes.get("low", [None])[i],
                "c": quotes.get("close", [None])[i],
                "vol": quotes.get("volume", [None])[i],
            })
        return candles
