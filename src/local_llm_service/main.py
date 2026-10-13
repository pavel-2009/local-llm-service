import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic, perf_counter
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .tools import get_weather


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)

MAX_CONCURRENT_REQUESTS = int(
    os.getenv("MAX_CONCURRENT_REQUESTS", "2")
)

OLLAMA_RETRIES = max(0, int(os.getenv("OLLAMA_RETRIES", "1")))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "0.2"))
UPSTREAM_TOTAL_TIMEOUT = float(os.getenv("UPSTREAM_TOTAL_TIMEOUT", "150"))
RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "30")))
RATE_LIMIT_WINDOW_SECONDS = max(
    1.0,
    float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
)

CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "3"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "120"))
WRITE_TIMEOUT = float(os.getenv("WRITE_TIMEOUT", "10"))
POOL_TIMEOUT = float(os.getenv("POOL_TIMEOUT", "5"))

RATE_LIMITED_PATHS = {"/generate", "/stream", "/extract", "/weather"}

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Pydantic models
# --------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    model: str


class SupportClassification(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    category: Literal[
        "account_access",
        "billing",
        "technical_issue",
        "other",
    ]

    priority: Literal["low", "medium", "high"]

    summary: str = Field(min_length=1, max_length=300)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class WeatherToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1, max_length=200)


# --------------------------------------------------
# Application lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    )

    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT_REQUESTS,
        max_keepalive_connections=MAX_CONCURRENT_REQUESTS,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:
        app.state.http_client = client
        app.state.llm_semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_REQUESTS
        )
        app.state.rate_limit_lock = asyncio.Lock()
        app.state.rate_limit_hits = {}

        logger.info("LLM service started; model=%s", OLLAMA_MODEL)

        try:
            yield
        finally:
            logger.info("LLM service shutting down")


app = FastAPI(
    title="Local LLM service",
    lifespan=lifespan,
)


@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    if request.method != "POST" or request.url.path not in RATE_LIMITED_PATHS:
        return await call_next(request)

    client_host = request.client.host if request.client else "unknown"
    now = monotonic()
    retry_after = 0

    async with request.app.state.rate_limit_lock:
        hits = request.app.state.rate_limit_hits
        window_start, count = hits.get(client_host, (now, 0))
        if now - window_start >= RATE_LIMIT_WINDOW_SECONDS:
            window_start, count = now, 0

        if count >= RATE_LIMIT_REQUESTS:
            retry_after = max(
                1,
                math.ceil(RATE_LIMIT_WINDOW_SECONDS - (now - window_start)),
            )
        else:
            hits[client_host] = (window_start, count + 1)

        if len(hits) > 10_000:
            expired_before = now - RATE_LIMIT_WINDOW_SECONDS
            for host, (started, _) in list(hits.items()):
                if started < expired_before:
                    del hits[host]

    if retry_after:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


# --------------------------------------------------
# Shared helpers
# --------------------------------------------------

def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def get_llm_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.llm_semaphore


def build_payload(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    stream: bool = False,
    tools: list[dict] | None = None,
) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
        },
        "tools": tools or [],
    }


def raise_upstream_error(exc: Exception) -> None:
    """Translate errors from the Ollama HTTP request into API errors."""

    if isinstance(exc, httpx.TimeoutException):
        raise HTTPException(
            status_code=504,
            detail="Timed out waiting for Ollama",
        ) from exc

    if isinstance(exc, httpx.HTTPStatusError):
        logger.error(
            "Ollama returned HTTP %s",
            exc.response.status_code,
        )

        raise HTTPException(
            status_code=502,
            detail="Ollama returned an HTTP error",
        ) from exc

    if isinstance(exc, httpx.RequestError):
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama",
        ) from exc


def is_retryable_upstream_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.RequestError))


async def post_ollama_chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    payload: dict,
) -> httpx.Response:
    try:
        async with asyncio.timeout(UPSTREAM_TOTAL_TIMEOUT):
            for attempt in range(OLLAMA_RETRIES + 1):
                try:
                    async with semaphore:
                        response = await client.post(
                            f"{OLLAMA_URL}/api/chat",
                            json=payload,
                        )
                    response.raise_for_status()
                    return response
                except (httpx.TimeoutException,
                        httpx.HTTPStatusError,
                        httpx.RequestError) as exc:
                    if attempt >= OLLAMA_RETRIES or not is_retryable_upstream_error(exc):
                        raise
                    logger.warning(
                        "Retrying Ollama request; retry=%s",
                        attempt + 1,
                    )
                    await asyncio.sleep(
                        RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Timed out waiting for Ollama",
        ) from exc
    except (httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.RequestError) as exc:
        raise_upstream_error(exc)
        raise AssertionError("Unreachable")

    raise AssertionError("Unreachable")


async def call_ollama_chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    *,
    temperature: float,
    tools: list[dict] | None = None,
) -> dict:
    payload = build_payload(
        messages,
        temperature=temperature,
        tools=tools,
    )

    response = await post_ollama_chat(client, semaphore, payload)

    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from Ollama",
        ) from exc


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------
# Generate
# --------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest,
    request: Request,
) -> GenerateResponse:
    client = get_http_client(request)
    semaphore = get_llm_semaphore(request)

    payload = build_payload(
        [{"role": "user", "content": body.prompt}],
        temperature=body.temperature,
    )

    started_at = perf_counter()

    response = await post_ollama_chat(client, semaphore, payload)

    try:
        data = response.json()
        text = data["message"]["content"]

        if not isinstance(text, str):
            raise ValueError("Expected a string in message.content")

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from Ollama",
        ) from exc

    logger.info(
        "Generation completed; elapsed=%.2fs",
        perf_counter() - started_at,
    )

    return GenerateResponse(
        text=text,
        model=OLLAMA_MODEL,
    )


# --------------------------------------------------
# Streaming
# --------------------------------------------------

@app.post("/stream")
async def stream(
    body: GenerateRequest,
    request: Request,
) -> StreamingResponse:
    client = get_http_client(request)
    semaphore = get_llm_semaphore(request)

    payload = build_payload(
        [{"role": "user", "content": body.prompt}],
        temperature=body.temperature,
        stream=True,
    )

    async def generate_chunks() -> AsyncIterator[str]:
        started_at = perf_counter()
        first_chunk_logged = False
        received_upstream_event = False

        try:
            async with asyncio.timeout(UPSTREAM_TOTAL_TIMEOUT):
                for attempt in range(OLLAMA_RETRIES + 1):
                    try:
                        # Hold the slot for the entire upstream stream.
                        async with semaphore:
                            async with client.stream(
                                "POST",
                                f"{OLLAMA_URL}/api/chat",
                                json=payload,
                            ) as response:
                                response.raise_for_status()

                                async for line in response.aiter_lines():
                                    if not line:
                                        continue

                                    received_upstream_event = True
                                    data = json.loads(line)
                                    message = data.get("message", {})
                                    content = message.get("content", "")

                                    if content:
                                        if not first_chunk_logged:
                                            logger.info(
                                                "Stream TTFT=%.2fs",
                                                perf_counter() - started_at,
                                            )
                                            first_chunk_logged = True

                                        yield json.dumps({
                                            "type": "chunk",
                                            "text": content,
                                        }) + "\n"

                                    if data.get("done"):
                                        yield json.dumps({
                                            "type": "done",
                                        }) + "\n"
                                        logger.info(
                                            "Stream completed; elapsed=%.2fs",
                                            perf_counter() - started_at,
                                        )
                                        return
                        return
                    except (httpx.TimeoutException,
                            httpx.HTTPStatusError,
                            httpx.RequestError) as exc:
                        if (
                            received_upstream_event
                            or attempt >= OLLAMA_RETRIES
                            or not is_retryable_upstream_error(exc)
                        ):
                            raise
                        logger.warning(
                            "Retrying Ollama stream; retry=%s",
                            attempt + 1,
                        )
                        await asyncio.sleep(
                            RETRY_BACKOFF_SECONDS * (2 ** attempt)
                        )

        except TimeoutError:
            yield json.dumps({
                "type": "error",
                "message": "Timed out waiting for Ollama",
            }) + "\n"

        except httpx.TimeoutException:
            yield json.dumps({
                "type": "error",
                "message": "Timed out waiting for Ollama",
            }) + "\n"

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama streaming HTTP error: %s",
                exc.response.status_code,
            )

            yield json.dumps({
                "type": "error",
                "message": "Ollama returned an HTTP error",
                "status_code": exc.response.status_code,
            }) + "\n"

        except httpx.RequestError:
            yield json.dumps({
                "type": "error",
                "message": "Could not connect to Ollama",
            }) + "\n"

        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.exception("Invalid Ollama streaming response")

            yield json.dumps({
                "type": "error",
                "message": "Invalid streaming response from Ollama",
            }) + "\n"

    return StreamingResponse(
        generate_chunks(),
        media_type="application/x-ndjson",
    )


# --------------------------------------------------
# Structured extraction
# --------------------------------------------------

@app.post("/extract", response_model=SupportClassification)
async def extract(
    body: ExtractRequest,
    request: Request,
) -> SupportClassification:
    client = get_http_client(request)
    semaphore = get_llm_semaphore(request)

    messages = [
        {
            "role": "system",
            "content": (
                "Classify the customer support message. "
                "Return a category, priority and concise summary. "
                "Use only the categories and priorities allowed "
                "by the provided JSON schema."
            ),
        },
        {
            "role": "user",
            "content": body.text,
        },
    ]

    payload = build_payload(
        messages,
        temperature=0.0,
    )

    # Ollama supports constrained structured output via JSON Schema.
    payload["format"] = SupportClassification.model_json_schema()

    response = await post_ollama_chat(client, semaphore, payload)

    try:
        data = response.json()
        content = data["message"]["content"]

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty or invalid model output")

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from Ollama",
        ) from exc

    try:
        result = SupportClassification.model_validate_json(content)

    except ValidationError as exc:
        logger.warning("LLM returned output that failed validation")

        raise HTTPException(
            status_code=502,
            detail="LLM output does not match the expected schema",
        ) from exc

    return result


@app.post("/weather")
async def weather(
    body: GenerateRequest,
    request: Request,
) -> GenerateResponse:
    client = get_http_client(request)
    semaphore = get_llm_semaphore(request)
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful weather assistant. Use get_weather to "
                "retrieve current conditions before answering. Never invent "
                "weather data. Answer in the user's language and include "
                "the location and local time when available."
            ),
        },
        {"role": "user", "content": body.prompt},
    ]
    weather_tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, preferably with country.",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }]

    data = await call_ollama_chat(
        client,
        semaphore,
        messages,
        temperature=0.0,
        tools=weather_tools,
    )
    assistant_message = data.get("message")
    if not isinstance(assistant_message, dict):
        raise HTTPException(
            status_code=502,
            detail="Invalid response from Ollama",
        )

    tool_calls = assistant_message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise HTTPException(
            status_code=502,
            detail="The model did not request the weather tool",
        )
    if len(tool_calls) > 4:
        raise HTTPException(
            status_code=502,
            detail="The model requested too many tool calls",
        )

    messages.append(assistant_message)
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict) or function.get("name") != "get_weather":
            raise HTTPException(
                status_code=502,
                detail="The model requested an unsupported tool",
            )

        try:
            arguments = WeatherToolArguments.model_validate(
                function.get("arguments")
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail="The model returned invalid weather tool arguments",
            ) from exc

        try:
            weather_result = await get_weather(arguments.city, client)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail="Timed out waiting for the weather provider",
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Weather provider returned HTTP %s",
                exc.response.status_code,
            )
            raise HTTPException(
                status_code=502,
                detail="Weather provider returned an HTTP error",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail="Could not connect to the weather provider",
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception("Invalid response from weather provider")
            raise HTTPException(
                status_code=502,
                detail="Invalid response from the weather provider",
            ) from exc

        messages.append({
            "role": "tool",
            "content": json.dumps(weather_result, ensure_ascii=False),
        })

    final_data = await call_ollama_chat(
        client,
        semaphore,
        messages,
        temperature=0.0,
    )
    final_message = final_data.get("message")
    text = final_message.get("content") if isinstance(final_message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=502,
            detail="Invalid final response from Ollama",
        )

    return GenerateResponse(text=text, model=OLLAMA_MODEL)
