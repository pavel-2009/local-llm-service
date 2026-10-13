import httpx
import respx

OLLAMA_URL = "http://localhost:11434/api/chat"


@respx.mock
def test_stream_success(client):
    ollama_stream = (
        '{"message":{"role":"assistant","content":"Hello"},'
        '"done":false}\n'
        '{"message":{"role":"assistant","content":" world!"},'
        '"done":false}\n'
        '{"message":{"role":"assistant","content":""},'
        '"done":true}\n'
    )

    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            text=ollama_stream,
            headers={"Content-Type": "application/x-ndjson"},
        )
    )

    response = client.post(
        "/stream",
        json={"prompt": "Say hello"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/x-ndjson"
    )

    events = [
        __import__("json").loads(line)
        for line in response.text.splitlines()
        if line
    ]

    assert events == [
        {"type": "chunk", "text": "Hello"},
        {"type": "chunk", "text": " world!"},
        {"type": "done"},
    ]


@respx.mock
def test_stream_ollama_http_error(client):
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            500,
            json={"error": "Internal server error"},
        )
    )

    response = client.post(
        "/stream",
        json={"prompt": "Say hello"},
    )

    # StreamingResponse уже начал отправлять ответ.
    # Ошибка передаётся событием внутри потока.
    assert response.status_code == 200

    events = [
        __import__("json").loads(line)
        for line in response.text.splitlines()
        if line
    ]

    assert events == [
        {
            "type": "error",
            "message": "Ollama returned an HTTP error",
            "status_code": 500,
        }
    ]


@respx.mock
def test_stream_retries_temporary_error_before_first_event(client):
    responses = iter([
        httpx.Response(503, json={"error": "Temporarily unavailable"}),
        httpx.Response(
            200,
            text=(
                '{"message":{"role":"assistant","content":"Hello"},'
                '"done":false}\n'
                '{"message":{"role":"assistant","content":""},'
                '"done":true}\n'
            ),
            headers={"Content-Type": "application/x-ndjson"},
        ),
    ])
    route = respx.post(OLLAMA_URL).mock(
        side_effect=lambda _: next(responses)
    )

    response = client.post(
        "/stream",
        json={"prompt": "Say hello"},
    )

    events = [
        __import__("json").loads(line)
        for line in response.text.splitlines()
        if line
    ]
    assert response.status_code == 200
    assert route.call_count == 2
    assert events == [
        {"type": "chunk", "text": "Hello"},
        {"type": "done"},
    ]