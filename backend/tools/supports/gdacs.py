"""
GDACS (Global Disaster Alerting Coordination System) request builders and
response parsers.

What it does:
    Defines request specs and response parsers for the UN GDACS API.
    Fetches real-time disaster alerts covering earthquakes, floods, cyclones,
    volcanoes, wildfires, and droughts.  No authentication required.
    The SEARCH endpoint returns GeoJSON by default; the parse function
    handles both GeoJSON and RSS/XML fallback.

Entities in it:
    - BASE_URL: GDACS API root.
    - _normalize_limit: Coerces and clamps result count to integer bounds.
    - _parse_geojson_feature: Extracts structured data from a GeoJSON feature.
    - _parse_rss: Fallback parser for RSS/XML text responses.
    - Request/parse pair for: events.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://www.gdacs.org/Knowledge/overview.aspx
"""

from typing import Any
from xml.etree import ElementTree


BASE_URL = "https://www.gdacs.org/gdacsapi/api"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_limit(raw: Any, default: int = 25, lower: int = 1, upper: int = 100) -> int:
    """Coerce a result-count limit to an integer within bounds.

    Args:
        raw: Limit value from the LLM (int, float, or str).
        default: Fallback when raw is falsy or unparseable.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        Clamped integer limit.
    """
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return default
    return max(lower, min(value, upper))


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def events_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for current GDACS disaster alerts.

    Args:
        **kwargs: Generic LLM params.  Uses ``limit`` (coerced via
                  _normalize_limit).

    Returns:
        Request spec dict for http.fetch().
    """
    limit = _normalize_limit(kwargs.get("limit", 25))
    return {
        "path": "/events/geteventlist/SEARCH",
        "params": {"limit": limit},
        "timeout": 20.0,
        "follow_redirects": True,
        "response_format": "json",
    }


def events_parse(data: Any) -> list[dict[str, Any]]:
    """Parse GDACS events response (GeoJSON or fallback RSS/XML).

    The GDACS SEARCH endpoint normally returns GeoJSON.  If for any reason
    the response was decoded as a string (XML/RSS), falls back to RSS
    parsing.

    Args:
        data: Raw response data -- typically a dict (GeoJSON) or str (XML).

    Returns:
        List of event dicts with event_type, title, severity, country,
        coordinates, date.

    Raises:
        RuntimeError: If data cannot be parsed in either format.
    """
    if isinstance(data, dict):
        features = data.get("features", [])
        return [_parse_geojson_feature(feature) for feature in features]

    if isinstance(data, str):
        try:
            return _parse_rss(data)
        except ElementTree.ParseError as exc:
            raise RuntimeError(
                f"GDACS returned unparseable response: {exc}"
            ) from exc

    raise RuntimeError(f"Unexpected GDACS response type: {type(data)}")


def _parse_geojson_feature(feature: dict) -> dict[str, Any]:
    """Extract structured data from a single GeoJSON feature.

    Args:
        feature: A GeoJSON feature dict from the GDACS response.

    Returns:
        Normalized event dict.
    """
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


def _parse_rss(xml_text: str) -> list[dict[str, Any]]:
    """Fallback parser for RSS/XML GDACS responses.

    Args:
        xml_text: Raw XML text from the API.

    Returns:
        List of event dicts parsed from RSS items.
    """
    root = ElementTree.fromstring(xml_text)
    items = root.findall(".//item")
    results = []
    for item in items:
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
