"""Basic llm service"""

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
