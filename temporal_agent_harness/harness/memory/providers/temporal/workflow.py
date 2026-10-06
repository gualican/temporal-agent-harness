"""Long-running entity workflow holding one scope's memories (e.g. one user).

Design notes (the constraints that shaped this):

- Other workflows cannot signal/query/update this workflow directly; all access
  goes through the client-holding activities in `service.py`, started
  lazily via update-with-start so the entity exists on first write.
- Writes are *updates* (not signals): callers get validation and the stored
  item back. Update handlers may run activities, so embedding happens at write
  time (best-effort — a memory without a vector still works via keyword match).
- Reads are *queries*: synchronous, pure, and free — they append no history
  events. Query handlers cannot run activities, so semantic recall requires the
  caller to embed the query text and pass the vector in.
- Continue-as-new fires when Temporal suggests it (history growth) or after 7
  quiet days (so long-lived runs regularly restart onto current workflow code,
  keeping the patching window short). Handlers are drained first. State rides
  the CAN argument, so it must stay well under the 2MB payload limit —
  `MemoryState.max_memories` enforces eviction.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from temporal_agent_harness.harness.memory.providers.temporal.embeddings import embed_texts
from temporal_agent_harness.harness.memory.providers.temporal.models import (
    MemoryItem,
    MemoryState,
    RecallParams,
    RememberParams,
    ScoredMemory,
)
from temporal_agent_harness.harness.memory.providers.temporal.scoring import recall_memories

REFRESH_INTERVAL = timedelta(days=7)


def memory_workflow_id(scope: str) -> str:
    return f"memory-{scope}"


@workflow.defn
class MemoryWorkflow:
    @workflow.init
    def __init__(self, state: MemoryState) -> None:
        self.state = state

    @workflow.run
    async def run(self, state: MemoryState) -> None:
        try:
            await workflow.wait_condition(
                lambda: workflow.info().is_continue_as_new_suggested(),
                timeout=REFRESH_INTERVAL,
            )
        except asyncio.TimeoutError:
            pass  # periodic refresh even when quiet
        await workflow.wait_condition(workflow.all_handlers_finished)
        workflow.continue_as_new(self.state)

    @workflow.update
    async def remember(self, params: RememberParams) -> MemoryItem:
        vector: list[float] | None = None
        try:
            vectors = await workflow.execute_activity(
                embed_texts,
                [params.text],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            vector = vectors[0]
        except ActivityError:
            workflow.logger.warning("Embedding failed; storing memory without a vector")

        item = MemoryItem(
            id=str(workflow.uuid4()),
            text=params.text,
            tags=params.tags,
            created_at=workflow.now(),
            vector=vector,
        )
        self.state.memories.append(item)
        overflow = len(self.state.memories) - self.state.max_memories
        if overflow > 0:
            del self.state.memories[:overflow]
        return item.model_copy(update={"vector": None})

    @remember.validator
    def validate_remember(self, params: RememberParams) -> None:
        if not params.text.strip():
            raise ValueError("Memory text must not be empty")

    @workflow.update
    async def forget(self, memory_id: str) -> bool:
        before = len(self.state.memories)
        self.state.memories = [m for m in self.state.memories if m.id != memory_id]
        return len(self.state.memories) < before

    @workflow.query
    def recall(self, params: RecallParams) -> list[ScoredMemory]:
        return recall_memories(self.state.memories, params, workflow.now())

    @workflow.query
    def list_memories(self) -> list[MemoryItem]:
        return [m.model_copy(update={"vector": None}) for m in self.state.memories]
