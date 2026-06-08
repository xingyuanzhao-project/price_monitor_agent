"""
USGS Earthquake Hazards connector.

Fetches real-time earthquake data from the USGS GeoJSON feeds.
No authentication required. Updated every 5 minutes.

API base: https://earthquake.usgs.gov/earthquakes/feed/v1.0
Docs: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
"""

from typing import Any

import httpx

BASE_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


async def fetch_earthquakes(
    min_magnitude: str = "4.5",
    period: str = "week",
) -> list[dict[str, Any]]:
    """Fetch recent earthquakes above a magnitude threshold.

    Args:
        min_magnitude: Minimum magnitude feed -- "significant", "4.5", "2.5", "1.0", "all".
        period: Time window -- "hour", "day", "week", "month".

    Returns:
        List of earthquake dicts with place, magnitude, time, coordinates, depth.
    """
    url = f"{BASE_URL}/{min_magnitude}_{period}.geojson"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        features = data.get("features", [])
        return [
            {
                "place": feature["properties"].get("place", ""),
                "magnitude": feature["properties"].get("mag"),
                "time": feature["properties"].get("time"),
                "tsunami": feature["properties"].get("tsunami", 0),
                "alert": feature["properties"].get("alert"),
                "longitude": feature["geometry"]["coordinates"][0],
                "latitude": feature["geometry"]["coordinates"][1],
                "depth_km": feature["geometry"]["coordinates"][2],
                "url": feature["properties"].get("url", ""),
            }
            for feature in features
        ]
