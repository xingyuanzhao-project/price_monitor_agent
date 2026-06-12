"""
USGS Earthquake Hazards request builders and response parsers.

What it does:
    Defines request specs and response parsers for the USGS GeoJSON
    earthquake feeds.  Fetches real-time earthquake data updated every
    5 minutes.  No authentication required.

Entities in it:
    - BASE_URL: USGS earthquake feed summary root.
    - VALID_MAGNITUDES: Set of accepted magnitude threshold strings for
      USGS feed paths.
    - VALID_PERIODS: Set of accepted time-period strings for USGS feed
      paths.
    - _normalize_min_magnitude: Validates and normalizes earthquake
      magnitude threshold strings.
    - _normalize_period: Validates and normalizes time-period strings
      to USGS-compatible feed names.
    - Request/parse pair for: earthquakes.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
"""

from typing import Any


BASE_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

VALID_MAGNITUDES = {"significant", "4.5", "2.5", "1.0", "all"}

VALID_PERIODS = {"hour", "day", "week", "month"}

PERIOD_ALIASES: dict[str, str] = {
    "hourly": "hour", "1h": "hour", "1 hour": "hour",
    "daily": "day", "1d": "day", "1 day": "day", "today": "day",
    "weekly": "week", "1w": "week", "1 week": "week", "7d": "week",
    "monthly": "month", "1m": "month", "1 month": "month", "30d": "month",
}


def _normalize_min_magnitude(raw: Any) -> str:
    """Validate and normalize an earthquake magnitude threshold.

    The USGS feed uses predefined magnitude buckets: ``significant``,
    ``4.5``, ``2.5``, ``1.0``, ``all``.  Strips whitespace and falls back
    to ``"4.5"`` for unrecognized values.

    Args:
        raw: Magnitude threshold from the LLM.

    Returns:
        Valid USGS magnitude string.
    """
    cleaned = str(raw).strip().lower()
    if cleaned in VALID_MAGNITUDES:
        return cleaned
    return "4.5"


def _normalize_period(raw: Any) -> str:
    """Validate and normalize a time-period string for USGS feeds.

    Accepts common LLM variants like ``"weekly"``, ``"monthly"``,
    ``"1 day"``, etc.  Falls back to ``"week"`` for unrecognized values.

    Args:
        raw: Time period string from the LLM.

    Returns:
        Valid USGS period string (hour, day, week, or month).
    """
    cleaned = str(raw).strip().lower()
    if cleaned in VALID_PERIODS:
        return cleaned
    return PERIOD_ALIASES.get(cleaned, "week")


# ---------------------------------------------------------------------------
# earthquakes
# ---------------------------------------------------------------------------

def earthquakes_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for recent earthquakes above a magnitude threshold.

    The USGS feeds are pre-generated static GeoJSON files named by magnitude
    and time period, so the path encodes the query parameters directly.

    Args:
        **kwargs: Generic LLM params.  Uses ``min_magnitude``, ``period``.

    Returns:
        Request spec dict for http.fetch().
    """
    min_magnitude = _normalize_min_magnitude(kwargs.get("min_magnitude", "4.5"))
    period = _normalize_period(kwargs.get("period", "week"))
    return {"path": f"/{min_magnitude}_{period}.geojson"}


def earthquakes_parse(data: dict) -> list[dict[str, Any]]:
    """Parse USGS earthquake GeoJSON response.

    Args:
        data: Raw GeoJSON dict from the API.

    Returns:
        List of earthquake dicts with place, magnitude, time, coordinates,
        depth, tsunami alert status, and detail URL.
    """
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
