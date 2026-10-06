# ABOUTME: MemoryProvider backed by the Mem0 platform (https://mem0.ai) via mem0.AsyncMemoryClient
# (mem0ai 2.x, where identity goes in ``filters`` — the v3 API rejects it as a top-level argument).
# Needs the ``mem0`` extra and MEM0_API_KEY (or pass your own client, e.g. a self-hosted ``host=``).

from typing import Any

from mem0 import AsyncMemoryClient

from temporal_agent_harness.harness.memory.types import MemoryRecord


def _record(raw: dict[str, Any]) -> MemoryRecord:
    metadata = raw.get("metadata") or {}
    return MemoryRecord(
        id=raw["id"],
        text=raw["memory"],
        tags=list(metadata.get("tags", [])),
        score=raw.get("score"),
    )


class Mem0Provider:
    def __init__(self, client: AsyncMemoryClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> AsyncMemoryClient:
        # Lazy so constructing the provider needs no API key (tests, import-time wiring).
        if self._client is None:
            self._client = AsyncMemoryClient()  # reads MEM0_API_KEY
        return self._client

    async def remember(self, scope: str, text: str, tags: list[str]) -> list[MemoryRecord]:
        kwargs: dict[str, Any] = {"filters": {"user_id": scope}}
        if tags:
            kwargs["metadata"] = {"tags": tags}
        response = await self.client.add(text, **kwargs)
        # Mem0 extracts facts server-side: an add can yield several memories, or none (duplicate).
        return [_record(r) for r in response.get("results", []) if r.get("memory")]

    async def recall(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
        response = await self.client.search(query, filters={"user_id": scope}, top_k=limit)
        return [_record(r) for r in response.get("results", [])]

    async def list_all(self, scope: str) -> list[MemoryRecord]:
        response = await self.client.get_all(filters={"user_id": scope})
        return [_record(r) for r in response.get("results", [])]

    async def forget(self, scope: str, memory_id: str) -> bool:
        # Mem0 ids are global, so confirm the memory is in this scope before deleting.
        if memory_id not in {r.id for r in await self.list_all(scope)}:
            return False
        await self.client.delete(memory_id)
        return True
