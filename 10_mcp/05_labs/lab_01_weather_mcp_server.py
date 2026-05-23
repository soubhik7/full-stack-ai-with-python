"""
lab_01_weather_mcp_server.py — Weather MCP Server
==================================================
A real-world MCP server that fetches weather data from a free API
(no API key required — uses wttr.in and open-meteo.com).

Tools:
  get_current_weather(city)       → temperature, conditions, humidity
  get_forecast(city, days)        → multi-day forecast
  get_uv_index(city)              → UV index + safety recommendation
  compare_weather(city1, city2)   → side-by-side comparison

Run standalone:
    python 10_mcp/05_labs/lab_01_weather_mcp_server.py

Inspect interactively:
    mcp dev 10_mcp/05_labs/lab_01_weather_mcp_server.py
"""

import json
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_wttr(city: str, fmt: str = "j1") -> dict:
    """Fetch weather data from wttr.in (JSON API)."""
    url = f"https://wttr.in/{city.replace(' ', '+')}?format={fmt}"
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        if fmt == "j1":
            return resp.json()
        return {"text": resp.text}


UV_LEVELS = {
    range(0, 3): ("Low", "Minimal protection needed. Enjoy outdoors!"),
    range(3, 6): ("Moderate", "Wear sunscreen SPF 30+. Seek shade midday."),
    range(6, 8): ("High", "Wear SPF 50+, hat and UV-protective clothing."),
    range(8, 11): ("Very High", "Minimize exposure 10am–4pm. Apply SPF 50+."),
    range(11, 20): ("Extreme", "Avoid outdoor activity midday. Full protection required."),
}


def _uv_description(uv_index: int) -> tuple[str, str]:
    for uv_range, (level, advice) in UV_LEVELS.items():
        if uv_index in uv_range:
            return level, advice
    return "Extreme", "Avoid outdoor activity. Full protection required."


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_current_weather(city: str) -> str:
    """
    Get the current weather conditions for a city.

    Returns temperature (°C and °F), feels-like temperature, humidity,
    wind speed, visibility, and a description of conditions.

    Args:
        city: Name of the city (e.g. "London", "New Delhi", "New York").

    Returns:
        A formatted weather report for the city.
    """
    try:
        data = _fetch_wttr(city)
        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        area_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_c = current["FeelsLikeC"]
        feels_f = current["FeelsLikeF"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        visibility_km = current["visibility"]
        description = current["weatherDesc"][0]["value"]

        return json.dumps({
            "location": f"{area_name}, {country}",
            "conditions": description,
            "temperature": {"celsius": f"{temp_c}°C", "fahrenheit": f"{temp_f}°F"},
            "feels_like": {"celsius": f"{feels_c}°C", "fahrenheit": f"{feels_f}°F"},
            "humidity": f"{humidity}%",
            "wind": f"{wind_kmph} km/h {wind_dir}",
            "visibility": f"{visibility_km} km",
        }, indent=2)

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch weather for '{city}': {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format for '{city}': {e}")


@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """
    Get a multi-day weather forecast for a city.

    Args:
        city: Name of the city.
        days: Number of forecast days (1–3). Defaults to 3.

    Returns:
        JSON array with daily forecasts including high/low temperatures,
        conditions, sunrise/sunset, and chance of rain.
    """
    days = max(1, min(days, 3))  # clamp to 1–3

    try:
        data = _fetch_wttr(city)
        weather_days = data["weather"][:days]

        forecast = []
        for day in weather_days:
            hourly = day["hourly"]
            # pick descriptions from morning, noon, evening
            conditions = list({h["weatherDesc"][0]["value"] for h in hourly})

            forecast.append({
                "date": day["date"],
                "max_temp": {"celsius": f"{day['maxtempC']}°C", "fahrenheit": f"{day['maxtempF']}°F"},
                "min_temp": {"celsius": f"{day['mintempC']}°C", "fahrenheit": f"{day['mintempF']}°F"},
                "sunrise": day["astronomy"][0]["sunrise"],
                "sunset": day["astronomy"][0]["sunset"],
                "conditions": conditions,
                "avg_humidity": f"{sum(int(h['humidity']) for h in hourly) // len(hourly)}%",
                "total_rainfall_mm": day.get("totalSnow_cm", "0"),
            })

        return json.dumps({
            "city": city,
            "forecast_days": days,
            "forecast": forecast,
        }, indent=2)

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch forecast for '{city}': {e}")


@mcp.tool()
def get_uv_index(city: str) -> str:
    """
    Get the current UV index for a city and safety recommendations.

    Args:
        city: Name of the city.

    Returns:
        UV index value, risk level, and sun protection recommendations.
    """
    try:
        data = _fetch_wttr(city)
        current = data["current_condition"][0]
        uv = int(current.get("uvIndex", 0))
        level, advice = _uv_description(uv)

        return json.dumps({
            "city": city,
            "uv_index": uv,
            "risk_level": level,
            "advice": advice,
            "max_safe_exposure_minutes": max(0, 60 - uv * 5),
        }, indent=2)

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch UV data for '{city}': {e}")


@mcp.tool()
def compare_weather(city1: str, city2: str) -> str:
    """
    Compare current weather between two cities.

    Args:
        city1: First city name.
        city2: Second city name.

    Returns:
        Side-by-side comparison of temperature, humidity, and conditions.
    """
    results = {}
    for city in [city1, city2]:
        data = _fetch_wttr(city)
        c = data["current_condition"][0]
        results[city] = {
            "temp_c": int(c["temp_C"]),
            "humidity": int(c["humidity"]),
            "conditions": c["weatherDesc"][0]["value"],
            "wind_kmph": int(c["windspeedKmph"]),
        }

    r1, r2 = results[city1], results[city2]
    warmer = city1 if r1["temp_c"] > r2["temp_c"] else city2
    diff = abs(r1["temp_c"] - r2["temp_c"])

    return json.dumps({
        city1: r1,
        city2: r2,
        "comparison": {
            "warmer_city": warmer,
            "temperature_difference": f"{diff}°C",
            "more_humid": city1 if r1["humidity"] > r2["humidity"] else city2,
        },
    }, indent=2)


if __name__ == "__main__":
    print("🌤️  Weather MCP Server starting...")
    print("Tools: get_current_weather, get_forecast, get_uv_index, compare_weather")
    print("No API key required — uses wttr.in")
    mcp.run()
