# ABOUTME: The MemoryProvider protocol — the one seam between the agent-facing memory tools and a
# memory backend (Mem0, a Temporal entity workflow, a vector DB, ...). Implement these four methods
# and every harness agent gets remember/recall/list/forget tools over it.

from typing import Protocol, runtime_checkable

from temporal_agent_harness.harness.memory.types import MemoryRecord


@runtime_checkable
class MemoryProvider(Protocol):
    """A scope-partitioned store of durable memories.

    ``scope`` is the partition key (a user id, a team, a project) and is ALWAYS supplied by the
    workflow via ``Injected`` — never by the model. Contract every implementation must keep:

    * Isolation: nothing written under one scope is ever returned, listed or deleted under another.
    * ``forget`` is scoped too: an id that exists but belongs to a different scope returns ``False``
      and deletes nothing (ids are model-supplied, and some backends' ids are global).
    * Methods run inside Temporal activities, so they may do network I/O and are retried on
      failure; make writes safe to repeat where the backend allows.
    """

    async def remember(self, scope: str, text: str, tags: list[str]) -> list[MemoryRecord]:
        """Store ``text``. Returns the record(s) stored — possibly several (or none) when the
        backend extracts/dedupes facts itself."""
        ...

    async def recall(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
        """The ``limit`` memories most relevant to ``query``, best first, with ``score`` set."""
        ...

    async def list_all(self, scope: str) -> list[MemoryRecord]:
        """Every memory in ``scope``."""
        ...

    async def forget(self, scope: str, memory_id: str) -> bool:
        """Delete one memory. ``True`` if it existed in ``scope`` and was deleted."""
        ...
