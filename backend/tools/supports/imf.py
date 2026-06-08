"""
IMF DataMapper connector.

Fetches macroeconomic indicators from the IMF's public JSON REST API.
No authentication required. Covers ~190 countries.

API base: https://www.imf.org/external/datamapper/api/v1
Docs: https://datahelp.imf.org/knowledgebase/articles/667681
"""

from typing import Any

import httpx

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"


async def fetch_indicator(
    indicator: str = "NGDP_RPCH",
    countries: str = "USA",
    periods: str = "",
) -> dict[str, Any]:
    """Fetch an IMF indicator for specified countries.

    Args:
        indicator: IMF indicator code. Common codes:
            NGDP_RPCH (GDP growth %), PCPIPCH (inflation %),
            LUR (unemployment rate), BCA_NGDPD (current account % GDP),
            GGXWDG_NGDP (government gross debt % GDP).
        countries: Comma-separated ISO 3-letter country codes (e.g. "USA,GBR,CHN").
        periods: Comma-separated years (e.g. "2023,2024,2025"). Empty = all available.

    Returns:
        Dict with indicator metadata and values keyed by country -> year.
    """
    url = f"{BASE_URL}/{indicator}"
    if countries:
        country_path = "/".join(c.strip() for c in countries.split(","))
        url += f"/{country_path}"
    params = {}
    if periods:
        params["periods"] = periods
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        values = data.get("values", {}).get(indicator, {})
        return {
            "indicator": indicator,
            "data": values,
        }


async def list_indicators() -> list[dict[str, Any]]:
    """List all available IMF DataMapper indicators.

    Returns:
        List of dicts with indicator code, label, and description.
    """
    url = f"{BASE_URL}/indicators"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        indicators = data.get("indicators", {})
        return [
            {"code": code, "label": meta.get("label", ""), "description": meta.get("description", "")}
            for code, meta in indicators.items()
        ]
