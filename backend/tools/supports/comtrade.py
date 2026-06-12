"""
UN Comtrade request builders and response parsers.

What it does:
    Defines request specs and response parsers for the UN Comtrade public
    preview API.  Fetches global merchandise trade flow data (imports/exports)
    between countries by HS commodity code.  No authentication required
    (preview endpoint limited to 500 records).

Entities in it:
    - BASE_URL: UN Comtrade public preview API root.
    - _normalize_country_code: Strips and validates UN M49 numeric country
      codes.
    - _normalize_hs_code: Strips whitespace and uppercases HS commodity
      codes.
    - _normalize_period: Strips and validates trade period strings.
    - _normalize_flow_code: Normalizes trade flow direction codes to
      uppercase.
    - Request/parse pairs for: trade.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://comtradeapi.un.org/
"""

from typing import Any


BASE_URL = "https://comtradeapi.un.org/public/v1/preview"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_country_code(raw: Any) -> str:
    """Strip and validate a UN M49 numeric country code.

    Removes whitespace and non-digit characters.  Falls back to ``"0"``
    (world aggregate) if the result is not a valid numeric string.

    Args:
        raw: Country code from the LLM (int or str).

    Returns:
        Clean numeric string for the Comtrade API.
    """
    cleaned = str(raw).strip()
    digits = "".join(c for c in cleaned if c.isdigit())
    return digits if digits else "0"


def _normalize_hs_code(raw: str) -> str:
    """Strip whitespace and uppercase an HS commodity code.

    Args:
        raw: HS code or ``"TOTAL"`` from the LLM.

    Returns:
        Trimmed, uppercased commodity code string.
    """
    return raw.strip().upper()


def _normalize_period(raw: Any) -> str:
    """Strip and validate a Comtrade trade period string.

    Accepts 4-digit (annual) or 6-digit (monthly YYYYMM) formats.
    Removes whitespace and separator characters.

    Args:
        raw: Period string from the LLM.

    Returns:
        Clean period string for the Comtrade API.
    """
    return str(raw).strip().replace("-", "").replace("/", "")


def _normalize_flow_code(raw: str) -> str:
    """Normalize a trade flow direction code to uppercase.

    Accepted codes are ``M`` (imports) and ``X`` (exports).

    Args:
        raw: Flow code from the LLM.

    Returns:
        Uppercase flow code string.
    """
    return raw.strip().upper()


# ---------------------------------------------------------------------------
# trade
# ---------------------------------------------------------------------------

def trade_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for merchandise trade flows between countries.

    Args:
        **kwargs: Generic LLM params.  Uses ``reporter_code`` (UN M49 numeric
                  string, e.g. "842" for USA), ``period`` (4- or 6-digit
                  year/month), ``partner_code``, ``flow_code`` (M/X),
                  ``commodity_code`` (HS code or "TOTAL").

    Returns:
        Request spec dict for http.fetch().
    """
    reporter_code = _normalize_country_code(kwargs.get("reporter_code", "842"))
    period = _normalize_period(kwargs.get("period", "2024"))
    partner_code = _normalize_country_code(kwargs.get("partner_code", "0"))
    flow_code = _normalize_flow_code(kwargs.get("flow_code", "M"))
    commodity_code = _normalize_hs_code(kwargs.get("commodity_code", "TOTAL"))

    return {
        "path": "/C/A/HS",
        "params": {
            "reporterCode": reporter_code,
            "period": period,
            "partnerCode": partner_code,
            "flowCode": flow_code,
            "cmdCode": commodity_code,
        },
        "timeout": 20.0,
    }


def trade_parse(data: dict) -> list[dict[str, Any]]:
    """Parse UN Comtrade trade data JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of trade flow dicts with reporter, partner, flow, commodity,
        trade value, net weight, and period.
    """
    records = data.get("data", [])
    return [
        {
            "reporter": record.get("reporterDesc", ""),
            "partner": record.get("partnerDesc", ""),
            "flow": record.get("flowDesc", ""),
            "commodity": record.get("cmdDesc", ""),
            "commodity_code": record.get("cmdCode", ""),
            "trade_value": record.get("primaryValue"),
            "net_weight_kg": record.get("netWgt"),
            "period": record.get("period"),
        }
        for record in records
    ]
