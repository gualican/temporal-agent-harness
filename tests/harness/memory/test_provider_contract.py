# ABOUTME: Contract suite every MemoryProvider must pass — run here against the bundled backends
# (the Mem0 one over an in-memory fake of its client). To check your own backend, add a factory to
# PROVIDERS; the properties below (isolation, scoped forget, ranking, limit) are the protocol's.

import uuid

import pytest
import pytest_asyncio
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_agent_harness.harness.memory import MemoryProvider
from temporal_agent_harness.harness.memory.providers.in_memory import InMemoryProvider
from temporal_agent_harness.harness.memory.providers.mem0 import Mem0Provider
from temporal_agent_harness.harness.memory.providers.temporal.embeddings import embed_texts
from temporal_agent_harness.harness.memory.providers.temporal.provider import (
    TemporalMemoryProvider,
)
from temporal_agent_harness.harness.memory.providers.temporal.service import MemoryService
from temporal_agent_harness.harness.memory.providers.temporal.workflow import MemoryWorkflow


class FakeMem0Client:
    """Just enough of mem0's AsyncMemoryClient (identity lives in ``filters``)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def add(self, messages, **kw):
        mid = f"m{len(self.rows) + 1}"
        self.rows[mid] = {
            "id": mid,
            "memory": messages,
            "user_id": kw["filters"]["user_id"],
            "metadata": kw.get("metadata"),
        }
        return {"results": [self.rows[mid]]}

    def _mine(self, kw):
        return [r for r in self.rows.values() if r["user_id"] == kw["filters"]["user_id"]]

    async def search(self, query, top_k=10, **kw):
        words = set(query.lower().split())
        hits = [
            {**r, "score": len(words & set(r["memory"].lower().split())) / len(words)}
            for r in self._mine(kw)
        ]
        hits = [h for h in hits if h["score"]]
        return {"results": sorted(hits, key=lambda h: -h["score"])[:top_k]}

    async def get_all(self, **kw):
        return {"results": self._mine(kw)}

    async def delete(self, memory_id, **kw):
        del self.rows[memory_id]


PROVIDERS = ["in_memory", "mem0", "temporal"]


@pytest_asyncio.fixture(params=PROVIDERS)
async def provider(request) -> MemoryProvider:
    if request.param == "in_memory":
        yield InMemoryProvider()
    elif request.param == "mem0":
        yield Mem0Provider(FakeMem0Client())
    else:
        # Real entity workflows on a time-skipping server. No OPENAI_API_KEY here, so embedding
        # fails best-effort and recall exercises the keyword-scoring fallback.
        env = await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        )
        queue = f"memory-contract-{uuid.uuid4()}"
        try:
            async with Worker(
                env.client, task_queue=queue, workflows=[MemoryWorkflow], activities=[embed_texts]
            ):
                yield TemporalMemoryProvider(MemoryService(env.client, task_queue=queue))
        finally:
            await env.shutdown()


async def test_isolation_between_scopes(provider):
    await provider.remember("alice", "likes vim editor", [])
    await provider.remember("bob", "likes emacs editor", [])
    assert [r.text for r in await provider.list_all("alice")] == ["likes vim editor"]
    # Ranking backends may return weak matches, but never another scope's memories.
    assert all("emacs" not in r.text for r in await provider.recall("alice", "emacs", 5))


async def test_recall_ranks_scores_and_limits(provider):
    await provider.remember("a", "works on temporal workflows", [])
    await provider.remember("a", "temporal workflows and agents daily", [])
    await provider.remember("a", "unrelated gardening", [])
    hits = await provider.recall("a", "temporal workflows agents", 1)
    assert len(hits) == 1 and hits[0].text.startswith("temporal workflows and agents")
    assert hits[0].score is not None


async def test_forget_is_scoped_and_reports_result(provider):
    [mine] = await provider.remember("alice", "likes vim", [])
    [theirs] = await provider.remember("bob", "likes emacs", [])
    assert await provider.forget("alice", theirs.id) is False  # someone else's id: no-op
    assert len(await provider.list_all("bob")) == 1
    assert await provider.forget("alice", mine.id) is True
    assert await provider.forget("alice", mine.id) is False  # already gone
    assert await provider.list_all("alice") == []
