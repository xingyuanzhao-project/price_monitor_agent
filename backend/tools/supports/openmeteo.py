"""
Open-Meteo weather and climate connector.

Fetches weather forecasts and historical climate data from the Open-Meteo API.
Processes Copernicus ERA5 reanalysis. No authentication required. No rate limits.

API base: https://api.open-meteo.com/v1
Docs: https://open-meteo.com/en/docs
"""

from typing import Any

import httpx

BASE_URL = "https://api.open-meteo.com/v1"


async def fetch_forecast(
    latitude: float,
    longitude: float,
    daily: str = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
    forecast_days: int = 7,
) -> dict[str, Any]:
    """Fetch weather forecast for a location.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        daily: Comma-separated daily variables.
        forecast_days: Number of days to forecast (1-16).

    Returns:
        Dict with daily arrays keyed by variable name.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily,
        "forecast_days": min(forecast_days, 16),
        "timezone": "UTC",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{BASE_URL}/forecast", params=params)
        response.raise_for_status()
        return response.json()


async def fetch_historical(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    daily: str = "temperature_2m_max,temperature_2m_min,precipitation_sum",
) -> dict[str, Any]:
    """Fetch historical weather data (ERA5 reanalysis).

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        daily: Comma-separated daily variables.

    Returns:
        Dict with daily arrays keyed by variable name.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily,
        "timezone": "UTC",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
