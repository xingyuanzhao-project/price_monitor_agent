"""
CoinGecko public API request builders and response parsers.

What it does:
    Defines request specs and response parsers for CoinGecko's free REST API v3.
    Covers simple prices, historical market charts, and trending coins.
    No authentication required for public endpoints.

Entities in it:
    - BASE_URL: CoinGecko API v3 root.
    - _normalize_coin_id: Ensures coin ID is a lowercase slug.
    - _normalize_vs_currency: Ensures target currency is lowercase.
    - Request/parse pairs for: ticker, ohlcv, trending.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://docs.coingecko.com/reference/introduction
"""

from typing import Any


BASE_URL = "https://api.coingecko.com/api/v3"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_coin_id(raw: str) -> str:
    """Ensure coin ID is a lowercase slug as required by CoinGecko.

    CoinGecko expects lowercase slug identifiers like ``bitcoin``, ``ethereum``.

    Args:
        raw: Coin identifier from the LLM.

    Returns:
        Lowercase stripped coin ID.
    """
    return raw.strip().lower()


def _normalize_vs_currency(raw: str) -> str:
    """Ensure target currency is lowercase as required by CoinGecko.

    Args:
        raw: Currency string from the LLM (e.g. "USD", "eur").

    Returns:
        Lowercase currency string.
    """
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# ticker (simple price)
# ---------------------------------------------------------------------------

def ticker_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a coin's simple price.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (as coin_id),
                  ``vs_currency``.

    Returns:
        Request spec dict for http.fetch().
    """
    coin_id = _normalize_coin_id(kwargs.get("symbol", "bitcoin"))
    vs_currency = _normalize_vs_currency(kwargs.get("vs_currency", "usd"))
    return {
        "path": "/simple/price",
        "params": {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        },
    }


def ticker_parse(data: dict) -> dict[str, Any]:
    """Parse CoinGecko simple price JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with coin_id, price, market_cap, vol_24h, change_24h.
    """
    if not data:
        return {}
    coin_id = next(iter(data))
    coin_data = data[coin_id]
    vs_currency = next(
        (k for k in coin_data if not k.endswith(("_market_cap", "_24h_vol", "_24h_change"))),
        "",
    )
    return {
        "coin_id": coin_id,
        "price": coin_data.get(vs_currency),
        "market_cap": coin_data.get(f"{vs_currency}_market_cap"),
        "vol_24h": coin_data.get(f"{vs_currency}_24h_vol"),
        "change_24h": coin_data.get(f"{vs_currency}_24h_change"),
    }


# ---------------------------------------------------------------------------
# ohlcv (market chart)
# ---------------------------------------------------------------------------

def ohlcv_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for historical market chart data.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol`` (as coin_id),
                  ``vs_currency``, ``days``.

    Returns:
        Request spec dict for http.fetch().
    """
    coin_id = _normalize_coin_id(kwargs.get("symbol", "bitcoin"))
    vs_currency = _normalize_vs_currency(kwargs.get("vs_currency", "usd"))
    days = str(kwargs.get("days", 7))
    return {
        "path": f"/coins/{coin_id}/market_chart",
        "params": {"vs_currency": vs_currency, "days": days},
    }


def ohlcv_parse(data: dict) -> dict[str, list]:
    """Parse CoinGecko market chart JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with prices, market_caps, total_volumes — each a list of [ts, value].
    """
    return {
        "prices": data.get("prices", []),
        "market_caps": data.get("market_caps", []),
        "total_volumes": data.get("total_volumes", []),
    }


# ---------------------------------------------------------------------------
# trending
# ---------------------------------------------------------------------------

def trending_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for trending coins.

    Args:
        **kwargs: Generic LLM params.  No params needed for this endpoint.

    Returns:
        Request spec dict for http.fetch().
    """
    return {"path": "/search/trending", "params": {}}


def trending_parse(data: dict) -> list[dict[str, Any]]:
    """Parse CoinGecko trending coins JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of trending coin dicts with id, name, symbol, market_cap_rank.
    """
    coins = data.get("coins", [])
    return [
        {
            "id": c["item"]["id"],
            "name": c["item"]["name"],
            "symbol": c["item"]["symbol"],
            "market_cap_rank": c["item"].get("market_cap_rank"),
        }
        for c in coins
    ]
