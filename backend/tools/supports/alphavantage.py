"""
Alpha Vantage API request builders and response parsers.

What it does:
    Defines request specs and response parsers for Alpha Vantage's REST API.
    Covers global quotes, daily time series (OHLCV), and crypto exchange rates.
    Requires a free API key (passed as parameter).

Entities in it:
    - BASE_URL: Alpha Vantage query endpoint.
    - _normalize_symbol: Uppercases and strips the symbol.
    - Request/parse pairs for: ohlcv, quote, crypto.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://www.alphavantage.co/documentation/
"""

from typing import Any


BASE_URL = "https://www.alphavantage.co/query"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_symbol(raw: str) -> str:
    """Uppercase and strip the symbol for Alpha Vantage.

    Args:
        raw: Symbol string from the LLM.

    Returns:
        Uppercase stripped symbol.
    """
    return raw.upper().strip()


# ---------------------------------------------------------------------------
# quote (global quote)
# ---------------------------------------------------------------------------

def quote_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a global quote lookup.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    return {
        "path": "",
        "params": {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
    }


def quote_parse(data: dict) -> dict[str, Any]:
    """Parse Alpha Vantage global quote JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Normalized dict with symbol, price, change, change_pct, volume.
    """
    gq = data.get("Global Quote", {})
    return {
        "symbol": gq.get("01. symbol", ""),
        "price": gq.get("05. price"),
        "change": gq.get("09. change"),
        "change_pct": gq.get("10. change percent"),
        "volume": gq.get("06. volume"),
    }


# ---------------------------------------------------------------------------
# ohlcv (daily time series)
# ---------------------------------------------------------------------------

def ohlcv_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for daily OHLCV time series.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``api_key``,
                  ``outputsize``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    outputsize = kwargs.get("outputsize", "compact")
    return {
        "path": "",
        "params": {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": outputsize,
        },
    }


def ohlcv_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Alpha Vantage daily time series JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of daily bar dicts with date, o, h, l, c, vol.
    """
    ts = data.get("Time Series (Daily)", {})
    return [
        {
            "date": date,
            "o": values["1. open"],
            "h": values["2. high"],
            "l": values["3. low"],
            "c": values["4. close"],
            "vol": values["5. volume"],
        }
        for date, values in ts.items()
    ]


# ---------------------------------------------------------------------------
# crypto (exchange rate)
# ---------------------------------------------------------------------------

def crypto_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for real-time crypto exchange rate.

    Args:
        **kwargs: Generic LLM params.  Uses ``from_currency``, ``to_currency``,
                  ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    from_currency = _normalize_symbol(kwargs.get("from_currency", "BTC"))
    to_currency = _normalize_symbol(kwargs.get("to_currency", "USD"))
    api_key = kwargs.get("api_key", "")
    return {
        "path": "",
        "params": {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": api_key,
        },
    }


def crypto_parse(data: dict) -> dict[str, Any]:
    """Parse Alpha Vantage crypto exchange rate JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with from, to, rate, last_refreshed.
    """
    rate_data = data.get("Realtime Currency Exchange Rate", {})
    return {
        "from": rate_data.get("1. From_Currency Code"),
        "to": rate_data.get("3. To_Currency Code"),
        "rate": rate_data.get("5. Exchange Rate"),
        "last_refreshed": rate_data.get("6. Last Refreshed"),
    }
