import httpx
import respx
from fastapi.testclient import TestClient

from src.local_llm_service.main import app


client = TestClient(app)

OLLAMA_URL = "http://localhost:11434/api/chat"


@respx.mock
def test_extract_success():
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"category":"billing",'
                        '"priority":"high",'
                        '"summary":"The customer was charged twice."}'
                    ),
                },
                "done": True,
            },
        )
    )

    response = client.post(
        "/extract",
        json={
            "text": "I was charged twice for my subscription.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "category": "billing",
        "priority": "high",
        "summary": "The customer was charged twice.",
    }


@respx.mock
def test_extract_rejects_invalid_model_output():
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"category":"billing",'
                        '"priority":"extreme",'
                        '"summary":"The customer was charged twice."}'
                    ),
                },
                "done": True,
            },
        )
    )

    response = client.post(
        "/extract",
        json={
            "text": "I was charged twice.",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "LLM output does not match the expected schema",
    }


def test_extract_rejects_empty_text():
    response = client.post(
        "/extract",
        json={"text": ""},
    )

    assert response.status_code == 422