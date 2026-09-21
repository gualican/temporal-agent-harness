"""Worker for the auto-routing hello-world OpenAI Agents agent.

Run from the repo root with:
    uv run --group examples python -m examples.auto_routing_hello.worker

A copy of examples/openai_hello's worker with one difference: instead of letting
OpenAIAgentsPlugin construct and register its own (fixed) ModelActivity, this worker
registers a JevAutoModelActivity itself — so any agent that sets model=AUTO_MODEL gets
routed per-call by jev-model-router's Jev-based classifier instead of always calling one
model. See temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing for why this needs
register_activities=False plus manual registration, rather than a plugin parameter.

Env vars (set in .env.local — see .env.example):
    TEMPORAL_CONFIG_FILE / TEMPORAL_PROFILE   Temporal connection profile
    OPENAI_API_KEY                            required — the routed models are OpenAI models
    TYPESAFE_API_KEY                          required — Jev classifies each request
    AUTO_ROUTING_HELLO_TASK_QUEUE             task queue to poll (default: auto-routing-hello)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from temporal_agent_harness.ai_sdks.openai_agents import OpenAIAgentsPlugin
from temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing import JevAutoModelActivity

from .workflow import TASK_QUEUE, TIER_TO_MODEL, AutoRoutingHelloAgentWorkflow


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    task_queue = os.environ.get("AUTO_ROUTING_HELLO_TASK_QUEUE", TASK_QUEUE)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("error: OPENAI_API_KEY env var not set")
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("error: TYPESAFE_API_KEY env var not set")

    # register_activities=False: this worker registers the model activities itself (below),
    # via JevAutoModelActivity, instead of letting the plugin build its own fixed ModelActivity.
    plugin = OpenAIAgentsPlugin(register_activities=False)

    jev_activity = JevAutoModelActivity(
        tier_to_model=TIER_TO_MODEL,
        default_model=TIER_TO_MODEL["moderate"],
    )

    connect_config = ClientConfig.load_client_connect_config()
    client = await Client.connect(**connect_config, plugins=[plugin])

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[AutoRoutingHelloAgentWorkflow],
        activities=[
            jev_activity.invoke_model_activity,
            jev_activity.invoke_model_activity_streaming,
        ],
    )
    print(
        f"Auto-routing hello agent worker ready: "
        f"profile={os.environ.get('TEMPORAL_PROFILE', 'default')!r} "
        f"address={connect_config.get('target_host')} "
        f"namespace={connect_config.get('namespace')} "
        f"taskQueue={task_queue} "
        f"tierToModel={TIER_TO_MODEL}",
        flush=True,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
