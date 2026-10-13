import json

import httpx
import respx


OLLAMA_URL = "http://localhost:11434/api/chat"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@respx.mock
def test_weather_calls_tool_and_returns_model_answer(client):
    ollama_responses = iter([
        httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Berlin"},
                        },
                    }],
                },
                "done": True,
            },
        ),
        httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "В Берлине сейчас ясно, 18 °C.",
                },
                "done": True,
            },
        ),
    ])
    ollama_route = respx.post(OLLAMA_URL).mock(
        side_effect=lambda _: next(ollama_responses)
    )
    respx.get(GEOCODING_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{
                    "name": "Berlin",
                    "country": "Germany",
                    "latitude": 52.52,
                    "longitude": 13.405,
                }],
            },
        )
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-10-13T12:00",
                    "temperature_2m": 18.0,
                    "apparent_temperature": 17.2,
                    "relative_humidity_2m": 60,
                    "wind_speed_10m": 10.0,
                    "weather_code": 0,
                },
            },
        )
    )

    response = client.post(
        "/weather",
        json={"prompt": "Какая погода в Берлине?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "В Берлине сейчас ясно, 18 °C.",
        "model": "qwen3:8b",
    }
    assert ollama_route.call_count == 2

    first_payload = json.loads(ollama_route.calls[0].request.content)
    assert first_payload["tools"][0]["function"]["name"] == "get_weather"

    second_payload = json.loads(ollama_route.calls[1].request.content)
    tool_result = json.loads(second_payload["messages"][-1]["content"])
    assert second_payload["messages"][-1]["role"] == "tool"
    assert tool_result["temperature_c"] == 18.0
    assert tool_result["condition"] == "clear sky"


@respx.mock
def test_weather_reports_when_model_does_not_call_tool(client):
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "It is sunny.",
                },
                "done": True,
            },
        )
    )

    response = client.post(
        "/weather",
        json={"prompt": "Какая погода в Берлине?"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The model did not request the weather tool",
    }