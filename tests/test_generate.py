import httpx
import respx

from src.local_llm_service import main as service

OLLAMA_URL = "http://localhost:11434/api/chat"


@respx.mock
def test_generate_success(client):
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {
                    "role": "assistant",
                    "content": "Dependency injection is a design pattern.",
                },
                "done": True,
            },
        )
    )

    response = client.post(
        "/generate",
        json={
            "prompt": "Explain dependency injection.",
            "temperature": 0.3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "Dependency injection is a design pattern.",
        "model": "qwen3:8b",
    }


@respx.mock
def test_generate_ollama_http_error(client):
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            500,
            json={"error": "Internal server error"},
        )
    )

    response = client.post(
        "/generate",
        json={"prompt": "Hello"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Ollama returned an HTTP error",
    }


def test_generate_rejects_empty_prompt(client):
    response = client.post(
        "/generate",
        json={"prompt": ""},
    )

    assert response.status_code == 422


def test_generate_rejects_invalid_temperature(client):
    response = client.post(
        "/generate",
        json={
            "prompt": "Hello",
            "temperature": 3.0,
        },
    )

    assert response.status_code == 422


@respx.mock
def test_generate_retries_temporary_ollama_error(client):
    responses = iter([
        httpx.Response(503, json={"error": "Temporarily unavailable"}),
        httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Recovered after retry.",
                },
                "done": True,
            },
        ),
    ])
    route = respx.post(OLLAMA_URL).mock(
        side_effect=lambda _: next(responses)
    )

    response = client.post(
        "/generate",
        json={"prompt": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Recovered after retry."
    assert route.call_count == 2


@respx.mock
def test_generate_returns_429_when_rate_limit_is_exceeded(
    client,
    monkeypatch,
):
    monkeypatch.setattr(service, "RATE_LIMIT_REQUESTS", 1)
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Hello.",
                },
                "done": True,
            },
        )
    )

    first_response = client.post(
        "/generate",
        json={"prompt": "First request"},
    )
    second_response = client.post(
        "/generate",
        json={"prompt": "Second request"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.headers["retry-after"]