import httpx
import respx
from fastapi.testclient import TestClient

from src.local_llm_service.main import app


client = TestClient(app)

OLLAMA_URL = "http://localhost:11434/api/chat"


@respx.mock
def test_generate_success():
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
def test_generate_ollama_http_error():
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


def test_generate_rejects_empty_prompt():
    response = client.post(
        "/generate",
        json={"prompt": ""},
    )

    assert response.status_code == 422


def test_generate_rejects_invalid_temperature():
    response = client.post(
        "/generate",
        json={
            "prompt": "Hello",
            "temperature": 3.0,
        },
    )

    assert response.status_code == 422