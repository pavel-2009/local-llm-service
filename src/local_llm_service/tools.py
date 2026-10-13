"""Functions callable by the LLM."""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


async def get_weather(
    city: str,
    client: httpx.AsyncClient,
) -> dict[str, object]:
    geocoding_response = await client.get(
        GEOCODING_URL,
        params={"name": city, "count": 1},
    )
    geocoding_response.raise_for_status()
    locations = geocoding_response.json().get("results", [])

    if not locations:
        return {"error": f"No location found for '{city}'."}

    location = locations[0]
    forecast_response = await client.get(
        FORECAST_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,weather_code,wind_speed_10m"
            ),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        },
    )
    forecast_response.raise_for_status()
    current = forecast_response.json()["current"]

    location_name = ", ".join(
        value
        for value in (
            location.get("name"),
            location.get("admin1"),
            location.get("country"),
        )
        if value
    )

    return {
        "location": location_name,
        "local_time": current["time"],
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "humidity_percent": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "condition": WEATHER_CODES.get(
            current["weather_code"],
            "unknown",
        ),
    }
