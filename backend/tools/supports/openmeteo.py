"""
Open-Meteo weather and climate request builders and response parsers.

What it does:
    Defines request specs and response parsers for the Open-Meteo API.
    Covers weather forecasts and historical climate data (Copernicus ERA5
    reanalysis).  No authentication required, no rate limits.

Entities in it:
    - BASE_URL: Open-Meteo forecast API v1 root.
    - ARCHIVE_BASE_URL: Open-Meteo historical archive API v1 root
      (used by the historical endpoint; callers must pair historical_request
      with this base URL).
    - _normalize_latitude: Coerces latitude values to floats within valid
      geographic bounds.
    - _normalize_longitude: Coerces longitude values to floats within valid
      geographic bounds.
    - _normalize_date: Normalizes date strings to YYYY-MM-DD format.
    - _normalize_daily_variables: Strips whitespace from comma-separated
      weather variable name lists.
    - Request/parse pairs for: forecast, historical.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://open-meteo.com/en/docs
"""

from typing import Any


BASE_URL = "https://api.open-meteo.com/v1"
ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_latitude(raw: Any) -> float:
    """Coerce a latitude value to a float within geographic bounds.

    Clamps to the valid range [-90, 90].

    Args:
        raw: Latitude from the LLM (int, float, or str).

    Returns:
        Float latitude clamped to valid bounds.
    """
    try:
        value = float(str(raw).strip())
    except (ValueError, TypeError):
        return 0.0
    return max(-90.0, min(value, 90.0))


def _normalize_longitude(raw: Any) -> float:
    """Coerce a longitude value to a float within geographic bounds.

    Clamps to the valid range [-180, 180].

    Args:
        raw: Longitude from the LLM (int, float, or str).

    Returns:
        Float longitude clamped to valid bounds.
    """
    try:
        value = float(str(raw).strip())
    except (ValueError, TypeError):
        return 0.0
    return max(-180.0, min(value, 180.0))


def _normalize_date(raw: str) -> str:
    """Normalize a date string to YYYY-MM-DD format.

    Handles common LLM variants: YYYY/MM/DD, MM-DD-YYYY,
    dates with trailing whitespace.

    Args:
        raw: Date string from the LLM.

    Returns:
        Date in YYYY-MM-DD format.
    """
    s = raw.strip()
    s = s.replace("/", "-")
    parts = s.split("-")
    if len(parts) == 3 and len(parts[0]) <= 2 and len(parts[2]) == 4:
        s = f"{parts[2]}-{parts[0]}-{parts[1]}"
    return s


def _normalize_daily_variables(raw: str) -> str:
    """Strip whitespace from comma-separated weather variable names.

    LLMs may insert spaces around commas in variable lists
    (e.g. ``"temperature_2m_max , precipitation_sum"``).

    Args:
        raw: Comma-separated variable string from the LLM.

    Returns:
        Cleaned comma-separated string with no extraneous spaces.
    """
    return ",".join(part.strip() for part in raw.split(","))


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def forecast_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a weather forecast at a location.

    Args:
        **kwargs: Generic LLM params.  Uses ``latitude``, ``longitude``,
                  ``daily`` (comma-separated variable names),
                  ``forecast_days`` (1-16).

    Returns:
        Request spec dict for http.fetch().
    """
    latitude = _normalize_latitude(kwargs.get("latitude", 0.0))
    longitude = _normalize_longitude(kwargs.get("longitude", 0.0))
    daily = _normalize_daily_variables(kwargs.get(
        "daily",
        "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
    ))
    forecast_days = min(int(kwargs.get("forecast_days", 7)), 16)
    return {
        "path": "/forecast",
        "params": {
            "latitude": latitude,
            "longitude": longitude,
            "daily": daily,
            "forecast_days": forecast_days,
            "timezone": "UTC",
        },
    }


def forecast_parse(data: dict) -> dict[str, Any]:
    """Parse Open-Meteo forecast JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Full response dict with daily arrays keyed by variable name.
    """
    return data


# ---------------------------------------------------------------------------
# historical
# ---------------------------------------------------------------------------

def historical_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for historical weather data (ERA5 reanalysis).

    Uses ARCHIVE_BASE_URL via the ``base_url`` spec key so that
    http.fetch() targets the archive API instead of the forecast API.

    Args:
        **kwargs: Generic LLM params.  Uses ``latitude``, ``longitude``,
                  ``start_date`` (YYYY-MM-DD), ``end_date`` (YYYY-MM-DD),
                  ``daily`` (comma-separated variable names).

    Returns:
        Request spec dict for http.fetch().
    """
    latitude = _normalize_latitude(kwargs.get("latitude", 0.0))
    longitude = _normalize_longitude(kwargs.get("longitude", 0.0))
    start_date = _normalize_date(kwargs.get("start_date", ""))
    end_date = _normalize_date(kwargs.get("end_date", ""))
    daily = _normalize_daily_variables(kwargs.get(
        "daily",
        "temperature_2m_max,temperature_2m_min,precipitation_sum",
    ))
    return {
        "path": "/archive",
        "base_url": ARCHIVE_BASE_URL,
        "params": {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": daily,
            "timezone": "UTC",
        },
        "timeout": 20.0,
    }


def historical_parse(data: dict) -> dict[str, Any]:
    """Parse Open-Meteo historical weather JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Full response dict with daily arrays keyed by variable name.
    """
    return data
