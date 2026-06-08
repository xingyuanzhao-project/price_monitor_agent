"""
UN Comtrade connector.

Fetches global merchandise trade flow data from the UN Comtrade public API.
No authentication required for the preview endpoint (limited to 500 records).

API base: https://comtradeapi.un.org/public/v1
Docs: https://comtradeapi.un.org/
"""

from typing import Any

import httpx

BASE_URL = "https://comtradeapi.un.org/public/v1/preview"


async def fetch_trade_data(
    reporter_code: int = 842,
    period: int = 2024,
    partner_code: int = 0,
    flow_code: str = "M",
    commodity_code: str = "TOTAL",
) -> list[dict[str, Any]]:
    """Fetch merchandise trade flows between countries.

    Args:
        reporter_code: UN M49 reporter country code (842=USA, 156=China, 276=Germany, 826=UK).
        period: Year (e.g. 2024).
        partner_code: UN M49 partner code (0=World).
        flow_code: M (imports), X (exports), or empty for both.
        commodity_code: HS code or "TOTAL" for all commodities.

    Returns:
        List of trade flow dicts with reporter, partner, trade value, commodity.
    """
    url = f"{BASE_URL}/C/A/HS"
    params: dict[str, Any] = {
        "reporterCode": reporter_code,
        "period": period,
        "partnerCode": partner_code,
        "flowCode": flow_code,
        "cmdCode": commodity_code,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
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
