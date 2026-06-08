"""
The Hear multi-country headline aggregator connector.

Fetches current headlines from 12-39 sources per country with AI-generated
overviews. Covers ideological diversity across 20 countries.
No authentication required.

API base: https://www.thehear.org/api
Docs: https://www.thehear.org/api
"""

from typing import Any

import httpx

BASE_URL = "https://www.thehear.org/api"


async def fetch_country(
    country: str = "us",
) -> dict[str, Any]:
    """Fetch current headlines and overviews for a country.

    Args:
        country: Country slug -- us, uk, germany, france, spain, turkey,
                 ukraine, australia, brazil, canada, india, italy, japan,
                 mexico, netherlands, poland, portugal, south-korea, sweden.

    Returns:
        Dict with headlines list and AI-generated overviews.
    """
    url = f"{BASE_URL}/country-view/{country}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
