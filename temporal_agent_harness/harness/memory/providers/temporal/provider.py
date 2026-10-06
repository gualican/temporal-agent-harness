# ABOUTME: TemporalMemoryProvider — adapts MemoryService (the client over the per-scope
# MemoryWorkflow entities) to the backend-neutral MemoryProvider protocol. Isolation and scoped
# forget come from the design: ids are only ever looked up inside one scope's own workflow.
# Worker/activity side only (it holds a Temporal client); never imported by workflow code.

from temporal_agent_harness.harness.memory.providers.temporal.models import (
    MemoryForget,
    MemoryItem,
    MemoryRecall,
    MemoryWrite,
)
from temporal_agent_harness.harness.memory.providers.temporal.service import MemoryService
from temporal_agent_harness.harness.memory.types import MemoryRecord


def _record(item: MemoryItem, score: float | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=item.id, text=item.text, tags=item.tags, score=score, created_at=item.created_at
    )


class TemporalMemoryProvider:
    def __init__(self, service: MemoryService) -> None:
        self._service = service

    async def remember(self, scope: str, text: str, tags: list[str]) -> list[MemoryRecord]:
        item = await self._service.memory_remember(MemoryWrite(scope=scope, text=text, tags=tags))
        return [_record(item)]

    async def recall(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
        scored = await self._service.memory_recall(
            MemoryRecall(scope=scope, query=query, top_k=limit)
        )
        return [_record(s.memory, s.score) for s in scored]

    async def list_all(self, scope: str) -> list[MemoryRecord]:
        return [_record(item) for item in await self._service.memory_list(scope)]

    async def forget(self, scope: str, memory_id: str) -> bool:
        return await self._service.memory_forget(MemoryForget(scope=scope, memory_id=memory_id))
