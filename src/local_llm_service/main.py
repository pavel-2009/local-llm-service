import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError


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

CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 120.0
WRITE_TIMEOUT = 10.0
POOL_TIMEOUT = 5.0

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

        logger.info("LLM service started; model=%s", OLLAMA_MODEL)

        try:
            yield
        finally:
            logger.info("LLM service shutting down")


app = FastAPI(
    title="Local LLM service",
    lifespan=lifespan,
)


# --------------------------------------------------
# Shared helpers
# --------------------------------------------------

def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def get_llm_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.llm_semaphore


def build_payload(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    stream: bool = False,
) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
        },
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

    try:
        async with semaphore:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()

    except (httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.RequestError) as exc:
        raise_upstream_error(exc)
        raise AssertionError("Unreachable")

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

    try:
        async with semaphore:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()

    except (httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.RequestError) as exc:
        raise_upstream_error(exc)
        raise AssertionError("Unreachable")

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