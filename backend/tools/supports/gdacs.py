"""
GDACS (Global Disaster Alerting Coordination System) connector.

Fetches real-time disaster alerts from the UN GDACS API.
Covers earthquakes, floods, cyclones, volcanoes, wildfires, droughts.
No authentication required.

API base: https://www.gdacs.org/gdacsapi/api
Docs: https://www.gdacs.org/Knowledge/overview.aspx
"""

from typing import Any
from xml.etree import ElementTree

import httpx

BASE_URL = "https://www.gdacs.org/gdacsapi/api"


async def fetch_events(
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Fetch current GDACS disaster alerts.

    Args:
        limit: Max events to return.

    Returns:
        List of event dicts with type, title, severity, country, coordinates, date.
    """
    url = f"{BASE_URL}/events/geteventlist/SEARCH"
    params = {"limit": min(limit, 100)}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            data = response.json()
            features = data.get("features", [])
            return [_parse_geojson_feature(feature) for feature in features[:limit]]
        return _parse_rss(response.text, limit)


def _parse_geojson_feature(feature: dict) -> dict[str, Any]:
    properties = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates", [0, 0])
    return {
        "event_type": properties.get("eventtype", ""),
        "title": properties.get("name", "") or properties.get("htmldescription", ""),
        "severity": properties.get("alertlevel", ""),
        "alert_score": properties.get("alertscore"),
        "country": properties.get("country", ""),
        "date": properties.get("fromdate", ""),
        "longitude": coordinates[0] if len(coordinates) > 0 else None,
        "latitude": coordinates[1] if len(coordinates) > 1 else None,
        "url": properties.get("url", ""),
        "episode_id": properties.get("episodeid"),
    }


def _parse_rss(xml_text: str, limit: int) -> list[dict[str, Any]]:
    """Fallback parser for RSS/XML response."""
    root = ElementTree.fromstring(xml_text)
    items = root.findall(".//item")
    results = []
    for item in items[:limit]:
        title_element = item.find("title")
        link_element = item.find("link")
        date_element = item.find("pubDate")
        description_element = item.find("description")
        results.append({
            "event_type": "",
            "title": title_element.text.strip() if title_element is not None and title_element.text else "",
            "severity": "",
            "country": "",
            "date": date_element.text.strip() if date_element is not None and date_element.text else "",
            "url": link_element.text.strip() if link_element is not None and link_element.text else "",
            "description": (description_element.text.strip()[:500] if description_element is not None and description_element.text else ""),
        })
    return results
