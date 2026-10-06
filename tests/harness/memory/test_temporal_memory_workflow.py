# ABOUTME: Behavior specific to the Temporal-workflow memory backend that the provider contract
# can't express: state surviving continue-as-new after a quiet period (the entity workflow
# restarts itself weekly), and writes landing in a lazily-created per-scope workflow.
#
# Run with: uv run pytest tests/harness/memory/test_temporal_memory_workflow.py -v

import uuid
from datetime import timedelta

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_agent_harness.harness.memory.providers.temporal.embeddings import embed_texts
from temporal_agent_harness.harness.memory.providers.temporal.models import MemoryWrite
from temporal_agent_harness.harness.memory.providers.temporal.service import MemoryService
from temporal_agent_harness.harness.memory.providers.temporal.workflow import (
    MemoryWorkflow,
    memory_workflow_id,
)


async def test_memory_continues_as_new_after_quiet_period() -> None:
    scope = f"test-{uuid.uuid4().hex[:8]}"
    queue = f"memory-can-{uuid.uuid4()}"
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        service = MemoryService(env.client, task_queue=queue)
        async with Worker(
            env.client, task_queue=queue, workflows=[MemoryWorkflow], activities=[embed_texts]
        ):
            await service.memory_remember(MemoryWrite(scope=scope, text="a fact worth keeping"))
            handle = env.client.get_workflow_handle_for(
                MemoryWorkflow.run, memory_workflow_id(scope)
            )
            run_id_before = (await handle.describe()).run_id

            await env.sleep(timedelta(days=8))

            assert (await handle.describe()).run_id != run_id_before, (
                "expected continue-as-new after 7 quiet days"
            )
            # State rides the continue-as-new argument.
            assert [m.text for m in await service.memory_list(scope)] == ["a fact worth keeping"]
