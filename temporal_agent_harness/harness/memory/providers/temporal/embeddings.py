"""Embedding helpers. The activity is used by the memory workflow at write time; the raw helper is
also called directly from client-side code (the service's recall and the CLI), which is already
outside workflow context. Needs the ``openai`` package (the ``openai-agents`` extra); imported
lazily so the workflow sandbox never loads it. Embedding is best-effort everywhere: without a key,
writes store no vector and recall falls back to keyword matching."""

from __future__ import annotations

from temporalio import activity

EMBEDDING_MODEL = "text-embedding-3-small"


async def embed(texts: list[str]) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


@activity.defn
async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await embed(texts)
