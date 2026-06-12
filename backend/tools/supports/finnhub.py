"""
Finnhub API request builders and response parsers.

What it does:
    Defines request specs and response parsers for Finnhub's REST API v1.
    Covers real-time quotes, company news, and earnings surprises.
    Requires a free API key (passed as parameter via ``token``).

Entities in it:
    - BASE_URL: Finnhub API v1 root.
    - _normalize_symbol: Uppercases and strips the ticker symbol.
    - Request/parse pairs for: quote, news, earnings.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://finnhub.io/docs/api
"""

from typing import Any


BASE_URL = "https://finnhub.io/api/v1"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_symbol(raw: str) -> str:
    """Uppercase and strip the ticker symbol for Finnhub.

    Args:
        raw: Symbol string from the LLM.

    Returns:
        Uppercase stripped symbol.
    """
    return raw.upper().strip()


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------

def quote_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a real-time stock quote.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    return {
        "path": "/quote",
        "params": {"symbol": symbol, "token": api_key},
    }


def quote_parse(data: dict) -> dict[str, Any]:
    """Parse Finnhub quote JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with current, high, low, open, prev_close, change, change_pct.
    """
    return {
        "current": data.get("c"),
        "high": data.get("h"),
        "low": data.get("l"),
        "open": data.get("o"),
        "prev_close": data.get("pc"),
        "change": data.get("d"),
        "change_pct": data.get("dp"),
    }


# ---------------------------------------------------------------------------
# news (company news)
# ---------------------------------------------------------------------------

def news_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for company news articles.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``api_key``,
                  ``from_date``, ``to_date``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    from_date = kwargs.get("from_date", "")
    to_date = kwargs.get("to_date", "")
    return {
        "path": "/company-news",
        "params": {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": api_key,
        },
    }


def news_parse(data: list) -> list[dict[str, Any]]:
    """Parse Finnhub company news JSON response.

    Args:
        data: Raw JSON list from the API.

    Returns:
        List of article dicts with headline, source, url, datetime, summary.
    """
    return [
        {
            "headline": a.get("headline", ""),
            "source": a.get("source", ""),
            "url": a.get("url", ""),
            "datetime": a.get("datetime"),
            "summary": a.get("summary", "")[:500],
        }
        for a in data
    ]


# ---------------------------------------------------------------------------
# earnings
# ---------------------------------------------------------------------------

def earnings_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for earnings surprises.

    Args:
        **kwargs: Generic LLM params.  Uses ``symbol``, ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    symbol = _normalize_symbol(kwargs.get("symbol", ""))
    api_key = kwargs.get("api_key", "")
    return {
        "path": "/stock/earnings",
        "params": {"symbol": symbol, "token": api_key},
    }


def earnings_parse(data: list) -> list[dict[str, Any]]:
    """Parse Finnhub earnings JSON response.

    Args:
        data: Raw JSON list from the API.

    Returns:
        List of earnings dicts with period, actual, estimate, surprise, surprise_pct.
    """
    return [
        {
            "period": e.get("period"),
            "actual": e.get("actual"),
            "estimate": e.get("estimate"),
            "surprise": e.get("surprise"),
            "surprise_pct": e.get("surprisePercent"),
        }
        for e in data
    ]
