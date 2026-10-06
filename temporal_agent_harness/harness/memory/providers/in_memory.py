# ABOUTME: A dependency-free MemoryProvider holding memories in process memory with keyword
# scoring. For tests, local demos, and as the smallest reference implementation of the protocol.
# NOT durable: contents vanish with the worker process.

import itertools
from datetime import datetime, timezone

from temporal_agent_harness.harness.memory.types import MemoryRecord


class InMemoryProvider:
    def __init__(self) -> None:
        self._by_scope: dict[str, dict[str, MemoryRecord]] = {}
        self._ids = itertools.count(1)

    async def remember(self, scope: str, text: str, tags: list[str]) -> list[MemoryRecord]:
        record = MemoryRecord(
            id=f"mem-{next(self._ids)}",
            text=text,
            tags=tags,
            created_at=datetime.now(timezone.utc),
        )
        self._by_scope.setdefault(scope, {})[record.id] = record
        return [record]

    async def recall(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
        words = set(query.lower().split())
        scored = []
        for record in self._by_scope.get(scope, {}).values():
            overlap = len(words & set(record.text.lower().split()))
            if overlap:
                scored.append(record.model_copy(update={"score": overlap / len(words)}))
        scored.sort(key=lambda r: r.score or 0.0, reverse=True)
        return scored[:limit]

    async def list_all(self, scope: str) -> list[MemoryRecord]:
        return list(self._by_scope.get(scope, {}).values())

    async def forget(self, scope: str, memory_id: str) -> bool:
        return self._by_scope.get(scope, {}).pop(memory_id, None) is not None
