from __future__ import annotations

import asyncio

import httpx
import numpy as np


async def embed_texts_http_json(
    texts: list[str],
    *,
    url: str,
    api_key: str = "",
    field: str = "text",
    vector_key: str = "embedding",
    timeout: float = 120.0,
    max_concurrency: int = 8,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_one(client: httpx.AsyncClient, text: str) -> list[float]:
        async with semaphore:
            response = await client.post(url, json={field: text}, headers=headers)
            response.raise_for_status()
            payload = response.json()
            vector = payload.get(vector_key)
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Некорректный ответ эмбеддера: {payload!r}")
            return vector

    async with httpx.AsyncClient(timeout=timeout) as client:
        vectors = await asyncio.gather(*(embed_one(client, text) for text in texts))

    return np.array(vectors, dtype=np.float32)


def is_openai_embedding_url(url: str) -> bool:
    normalized = url.rstrip("/")
    return normalized.endswith("/v1")
