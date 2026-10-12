"""Basic llm service"""

import os
import json
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, ValidationError

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)

app = FastAPI(title='Local LLM service')


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    text: str
    model: str

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class SupportClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: Literal[
        "account_access",
        "billing",
        "technical_issue",
        "other",
    ]

    priority: Literal["low", "medium", "high"]

    summary: str = Field(min_length=1, max_length=300)


@app.post('/generate', response_model=GenerateResponse)
async def generate(
    request: GenerateRequest
) -> GenerateResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "messages" : [
            {
                "role": "user",
                "content": request.prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": request.temperature
        }
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=3.0)
        ) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload
            )

            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an HTTP error",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned invalid JSON",
        ) from exc

    message = data.get("message")

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from Ollama",
        )

    content = message.get("content")

    if not isinstance(content, str):
        raise HTTPException(
            status_code=502,
            detail="Ollama response has no text content",
        )   

    return GenerateResponse(
        text=content,
        model=data.get("model", OLLAMA_MODEL),
    )   


@app.post("/stream")
async def stream(
    request: GenerateRequest
) -> StreamingResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": request.prompt,
            }
        ],
        "stream": True,
        "options": {
            "temperature": request.temperature,
        },
    }

    async def generate_chunks():
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=3.0)
            ) as client:
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
                            yield json.dumps(
                                {
                                    "type": "chunk",
                                    "text": content,
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        if data.get("done"):
                            yield json.dumps(
                                {"type": "done"},
                                ensure_ascii=False,
                            ) + "\n"

                            return

        except httpx.TimeoutException:
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Ollama request timed out",
                }
            ) + "\n"

        except httpx.HTTPStatusError as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Ollama returned an HTTP error",
                    "status_code": exc.response.status_code,
                }
            ) + "\n"

        except httpx.RequestError:
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Cannot connect to Ollama",
                }
            ) + "\n"

        except (json.JSONDecodeError, TypeError):
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Invalid response from Ollama",
                }
            ) + "\n"

    return StreamingResponse(
        generate_chunks(),
        media_type="application/x-ndjson"
    )


@app.post("/extract")
async def extract(
    request: ExtractRequest
) -> SupportClassification:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify customer support messages. "
                    "Return a classification based only on the user's message. "
                    "Do not invent facts. "
                    "Choose 'other' if no category fits. "
                    "Choose the priority based on urgency and impact. "
                    "The summary must be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": request.text,
            },
        ],
        "stream": False,
        "format": SupportClassification.model_json_schema(),
        "options": {
            "temperature": 0,
        },
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=3.0)
        ) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an HTTP error",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned invalid JSON",
        ) from exc

    message = data.get("message")

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from Ollama",
        )

    content = message.get("content")

    if not isinstance(content, str) or not content:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned no structured content",
        )

    try:
        return SupportClassification.model_validate_json(content)

    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM output does not match the expected schema",
        ) from exc
