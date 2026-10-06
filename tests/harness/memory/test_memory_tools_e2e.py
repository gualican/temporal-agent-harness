# ABOUTME: End-to-end test of the shared memory tools with NO model: a model-free parent drives the
# real tools through run_tool against an InMemoryProvider, proving the dispatcher -> activity ->
# provider path, scope injection, and that forget_memory cannot delete another scope's memory.
#
# Run with: uv run pytest tests/harness/memory -v

import uuid

import pytest_asyncio
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_agent_harness.harness.agent_protocol import (
    SEND_AGENT_MESSAGE_UPDATE,
    TURN_EVENTS_TOPIC,
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AgentMessage,
    AgentMessageReply,
)
from temporal_agent_harness.harness.memory import MEMORY_TOOLS, configure_memory
from temporal_agent_harness.harness.memory.providers.in_memory import InMemoryProvider
from temporal_agent_harness.harness import agent

from ._memory_e2e_parent import MemoryE2EParentWorkflow


@pytest_asyncio.fixture
async def run_step():
    configure_memory(InMemoryProvider())
    env = await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)
    task_queue = f"memory-e2e-{uuid.uuid4()}"
    try:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[MemoryE2EParentWorkflow],
            activities=[agent.tool_activity(t) for t in MEMORY_TOOLS],
        ):
            handle = await env.client.start_workflow(
                MemoryE2EParentWorkflow.run,
                AgentConfig(),
                id=f"MemoryE2E-{uuid.uuid4()}",
                task_queue=task_queue,
            )
            stream = WorkflowStreamClient.create(env.client, handle.id)
            seen = 0

            async def step(tool: str, user_id: str, arg: str = "") -> str:
                nonlocal seen
                await handle.execute_update(
                    SEND_AGENT_MESSAGE_UPDATE,
                    AgentMessage(
                        type="step",
                        payload={"tool": tool, "user_id": user_id, "arg": arg},
                        expected_turn=seen + 1,
                    ),
                    result_type=AgentMessageReply,
                )
                # Replay the stream from the start; take the Nth reply.
                ends = 0
                async for item in stream.subscribe(
                    topics=[TURN_EVENTS_TOPIC], from_offset=0, result_type=AgentEvent
                ):
                    ev = item.data.event
                    if ev.type == AgentEventType.REPLY:
                        ends += 1
                        if ends == seen + 1:
                            seen += 1
                            return ev.output["text"]
                raise AssertionError("stream ended before the handler finished")

            yield step
    finally:
        configure_memory(None)
        await env.shutdown()


async def test_tools_round_trip_scoped_by_injection(run_step):
    step = run_step

    saved = await step("remember", "alice", "I prefer vim")
    assert "I prefer vim" in saved
    await step("remember", "bob", "I prefer emacs")

    assert "vim" in await step("recall", "alice", "vim")
    assert await step("recall", "alice", "emacs") == "No stored memories match."
    listing = await step("list", "alice")
    assert "vim" in listing and "emacs" not in listing

    bob_id = (await step("list", "bob")).split("[")[1].split("]")[0]
    assert await step("forget", "alice", bob_id) == "No memory with that id."
    assert "emacs" in await step("list", "bob")

    alice_id = listing.split("[")[1].split("]")[0]
    assert await step("forget", "alice", alice_id) == "Memory deleted."
    assert await step("list", "alice") == "No memories stored."
