"""Client-holding layer that mediates all access to the memory entity workflows.

Temporal workflows cannot signal/query/update other workflows directly, so callers — the memory
tool activities (via ``TemporalMemoryProvider``), the CLI, tests — reach a scope's entity workflow
through a ``Client`` held here. Writes use update-with-start so the entity workflow exists lazily
on first touch; reads are queries and return empty for scopes never written.

Worker/client side only — workflow code never imports this module.
"""

from __future__ import annotations

from temporalio.client import Client, WithStartWorkflowOperation
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.service import RPCError, RPCStatusCode

from temporal_agent_harness.harness.memory.providers.temporal.embeddings import embed
from temporal_agent_harness.harness.memory.providers.temporal.models import (
    MemoryForget,
    MemoryItem,
    MemoryRecall,
    MemoryState,
    MemoryWrite,
    RecallParams,
    RememberParams,
    ScoredMemory,
)
from temporal_agent_harness.harness.memory.providers.temporal.workflow import (
    MemoryWorkflow,
    memory_workflow_id,
)


DEFAULT_TASK_QUEUE = "harness-memory"


class MemoryService:
    def __init__(self, client: Client, task_queue: str = DEFAULT_TASK_QUEUE) -> None:
        self._client = client
        # The queue a worker running MemoryWorkflow polls; entity workflows are started on it.
        self._task_queue = task_queue

    def _with_start(self, scope: str) -> WithStartWorkflowOperation:
        return WithStartWorkflowOperation(
            MemoryWorkflow.run,
            MemoryState(scope=scope),
            id=memory_workflow_id(scope),
            task_queue=self._task_queue,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )

    async def memory_remember(self, params: MemoryWrite) -> MemoryItem:
        return await self._client.execute_update_with_start_workflow(
            MemoryWorkflow.remember,
            RememberParams(text=params.text, tags=params.tags),
            start_workflow_operation=self._with_start(params.scope),
        )

    async def memory_recall(self, params: MemoryRecall) -> list[ScoredMemory]:
        # Caller-side embedding: the workflow's query handler can't run
        # activities, so the query vector must arrive precomputed. Best-effort —
        # without an API key recall degrades to keyword matching.
        query_vector: list[float] | None = None
        try:
            query_vector = (await embed([params.query]))[0]
        except Exception:
            pass
        handle = self._client.get_workflow_handle_for(
            MemoryWorkflow.run, memory_workflow_id(params.scope)
        )
        try:
            return await handle.query(
                MemoryWorkflow.recall,
                RecallParams(
                    query_text=params.query, query_vector=query_vector, top_k=params.top_k
                ),
            )
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                return []
            raise

    async def memory_forget(self, params: MemoryForget) -> bool:
        return await self._client.execute_update_with_start_workflow(
            MemoryWorkflow.forget,
            params.memory_id,
            start_workflow_operation=self._with_start(params.scope),
        )

    async def memory_list(self, scope: str) -> list[MemoryItem]:
        handle = self._client.get_workflow_handle_for(
            MemoryWorkflow.run, memory_workflow_id(scope)
        )
        try:
            return await handle.query(MemoryWorkflow.list_memories)
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                return []
            raise
