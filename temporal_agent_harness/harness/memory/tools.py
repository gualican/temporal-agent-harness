# ABOUTME: Backend-agnostic memory tools for harness agents: remember/recall/list/forget as
# activity tools over whichever MemoryProvider the worker configured, plus a context activity for
# injecting top-k memories into a prompt. Workflow code imports only this module (no backend deps).
#
# Wiring:
#   worker:    configure_memory(MyProvider(...));  plugin = AgentHarnessPlugin(tools=MEMORY_TOOLS)
#              Worker(..., activities=MEMORY_ACTIVITIES)        # only if you use memory_context
#   workflow:  as_openai_agent_tools(runner, MEMORY_TOOLS, injections={"scope": user_id})
#              instructions += MEMORY_INSTRUCTIONS
#
# ``scope`` is Injected on every tool, so the model never sees or picks it. forget_memory is
# deliberately NOT inherently_safe: deletion goes through the session's tool-approval policy.
#
# No ``from __future__ import annotations`` — these types cross Temporal's data converter.

from datetime import timedelta

from pydantic import BaseModel
from temporalio import activity
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

from temporal_agent_harness.harness import agent
from temporal_agent_harness.harness.memory.provider import MemoryProvider
from temporal_agent_harness.harness.memory.types import MemoryRecord

MEMORY_TOOL_TIMEOUT = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy=RetryPolicy(maximum_attempts=3),
)

MEMORY_INSTRUCTIONS = """

## Memory
You have durable long-term memory about this user that persists across conversations:
- remember_memory: store a short, self-contained fact worth keeping (preferences, ongoing
  projects, corrections). Do this proactively when the user shares durable information.
- recall_memories: search memory whenever past context could help; check it before asking the
  user something they may already have told you.
- list_memories: list everything stored (for "what do you know about me?").
- forget_memory: delete a memory by id when asked, or when one is wrong or stale. The user may be
  asked to approve the deletion."""

_provider: MemoryProvider | None = None


def configure_memory(provider: MemoryProvider | None) -> None:
    """Set the process-wide provider the memory tool bodies use. Call once at worker startup (and
    from tests); the bodies run as plain activities with no other way to reach it."""
    global _provider
    _provider = provider


def get_memory_provider() -> MemoryProvider:
    if _provider is None:
        raise RuntimeError(
            "memory tool used before configure_memory(provider) was called on this worker"
        )
    return _provider


def format_records(records: list[MemoryRecord]) -> str:
    return "\n".join(f"- [{r.id}] {r.text}" for r in records)


@agent.activity_tool_defn(inherently_safe=True, activity_config=MEMORY_TOOL_TIMEOUT)
async def remember_memory(
    scope: agent.Injected[str], text: str, tags: list[str] | None = None
) -> str:
    """Store a durable memory about the user.

    Args:
        text: One self-contained sentence stating the fact to remember.
        tags: Optional short topic tags (e.g. ["preference", "project"]).
    """
    stored = await get_memory_provider().remember(scope, text, tags or [])
    if not stored:
        return "Nothing new to remember."
    if len(stored) == 1:
        return f"Stored memory {stored[0].id}: {stored[0].text}"
    # Backends that extract facts themselves (e.g. Mem0) can turn one input into several memories.
    return f"Stored {len(stored)} memories:\n" + format_records(stored)


@agent.activity_tool_defn(inherently_safe=True, activity_config=MEMORY_TOOL_TIMEOUT)
async def recall_memories(scope: agent.Injected[str], query: str, limit: int = 5) -> str:
    """Search the user's stored memories.

    Args:
        query: What you want to know, phrased with distinctive keywords.
        limit: Maximum number of memories to return.
    """
    found = await get_memory_provider().recall(scope, query, limit)
    return format_records(found) if found else "No stored memories match."


@agent.activity_tool_defn(inherently_safe=True, activity_config=MEMORY_TOOL_TIMEOUT)
async def list_memories(scope: agent.Injected[str]) -> str:
    """List every memory stored for the user."""
    everything = await get_memory_provider().list_all(scope)
    return format_records(everything) if everything else "No memories stored."


@agent.activity_tool_defn(activity_config=MEMORY_TOOL_TIMEOUT)
async def forget_memory(scope: agent.Injected[str], memory_id: str) -> str:
    """Permanently delete a stored memory by its id (as shown by recall / list).

    Args:
        memory_id: The id of the memory to delete.
    """
    removed = await get_memory_provider().forget(scope, memory_id)
    return "Memory deleted." if removed else "No memory with that id."


MEMORY_TOOLS = [remember_memory, recall_memories, list_memories, forget_memory]


class MemoryContextRequest(BaseModel):
    scope: str
    query: str
    limit: int = 5


@activity.defn(name="memory_context")
async def memory_context(request: MemoryContextRequest) -> str:
    """Prompt-ready block of the memories most relevant to ``request.query`` ('' if none), for
    workflows that inject recalled memories up front instead of waiting for the model to ask. Call
    it from the workflow with ``workflow.execute_activity("memory_context", request, ...)``."""
    found = await get_memory_provider().recall(request.scope, request.query, request.limit)
    if not found:
        return ""
    return "\n\n## Memories recalled for this request\n" + format_records(found)


MEMORY_ACTIVITIES = [memory_context]
