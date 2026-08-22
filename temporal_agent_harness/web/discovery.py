"""Shared session-discovery helpers for the harness's web API and CLI.

Kept free of FastAPI (unlike ``web.app``) so non-web consumers — e.g.
``temporal_agent_harness.cli`` — can import it without pulling in the ``ui`` extra.
"""

from __future__ import annotations

from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.service import RPCError, RPCStatusCode

from temporal_agent_harness.web.session_manager import AgentRegistry, Session

_DISCOVERY_LIMIT = 200


async def workflow_execution_status(
    client: Client, workflow_id: str
) -> WorkflowExecutionStatus | None:
    """The workflow's current status, or ``None`` if no such execution exists."""
    handle = client.get_workflow_handle(workflow_id)
    try:
        desc = await handle.describe()
    except RPCError as exc:
        if exc.status != RPCStatusCode.NOT_FOUND:
            raise
        return None
    return desc.status


async def discover_untracked_sessions(
    client: Client,
    registry: AgentRegistry,
    known_workflow_ids: set[str],
) -> list[Session]:
    """Find agent workflows already running in the namespace that a session manager
    didn't start itself (e.g. launched directly against a worker, or by another session
    manager or the CLI), so a client can list and attach to them too, not only ones it
    created via ``create_session``.
    """
    if not registry.agents:
        return []

    escaped_types = [agent.workflow_type.replace("'", "''") for agent in registry.agents]
    types_filter = " OR ".join(f"WorkflowType='{workflow_type}'" for workflow_type in escaped_types)
    query = f"ExecutionStatus='Running' AND ({types_filter})"

    discovered: list[Session] = []
    async for execution in client.list_workflows(query=query, limit=_DISCOVERY_LIMIT):
        if execution.id in known_workflow_ids:
            continue
        descriptor = registry.by_workflow_type(execution.workflow_type)
        if descriptor is None:
            continue
        discovered.append(
            Session(
                workflow_id=execution.id,
                created_at=execution.start_time.timestamp(),
                label=descriptor.label,
                agent_workflow_type=execution.workflow_type,
                is_discovered=True,
            )
        )
    return discovered
