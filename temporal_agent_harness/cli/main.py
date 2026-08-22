"""``harness`` — a terminal client for the packaged session manager and any registered agent.

Shares the SAME ``SessionManagerWorkflow`` the web UI (``temporal_agent_harness.web``) drives,
so a session started from one is listable/attachable from the other. Talks to any
``@agent.accepts`` handler by introspecting its input schema (``AgentClient.get_agent_interface``)
rather than assuming a fixed message shape, so it works against the bundled ``ask(TextMessage)``
convention as well as a custom handler with different field names.

    harness agents
    harness chat --agent ask-me-anything
    harness sessions
    harness chat --session <workflow-id>
    harness approve --session <workflow-id>
    harness close --session <workflow-id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowHandle,
)
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import WorkflowAlreadyStartedError

from temporal_agent_harness.harness.agent_client import (
    AgentBusyError,
    AgentClient,
    AgentTurnError,
    AgentTurnTimeout,
    StaleTurnError,
)
from temporal_agent_harness.harness.agent_protocol import (
    AcceptedFunction,
    AgentConfig,
    AgentEvent,
    AgentEventType,
)
from temporal_agent_harness.utils.large_payload import with_large_payload_offload
from temporal_agent_harness.web import (
    SESSION_MANAGER_ID,
    SESSION_MANAGER_TASK_QUEUE,
    AgentRegistry,
    CreateSessionRequest,
    Session,
    SessionManagerWorkflow,
    load_agent_registry,
)
from temporal_agent_harness.web.discovery import (
    discover_untracked_sessions,
    workflow_execution_status,
)

# ---------------------------------------------------------------------------
# Connection / session-manager plumbing
# ---------------------------------------------------------------------------


async def _connect(address: str, namespace: str) -> Client:
    return await Client.connect(
        address,
        namespace=namespace,
        data_converter=await with_large_payload_offload(pydantic_data_converter),
    )


async def _ensure_session_manager(
    client: Client,
    registry: AgentRegistry,
    *,
    manager_workflow_id: str,
    manager_task_queue: str,
) -> WorkflowHandle[Any, Any]:
    """Connect to the session manager if it's running, else start it (same convention as
    ``web.app``'s lifespan) — so ``harness`` works standalone (no `uv run agent-session-manager`
    running yet) as well as alongside a running web UI."""
    status = await workflow_execution_status(client, manager_workflow_id)
    if status == WorkflowExecutionStatus.RUNNING:
        return client.get_workflow_handle(manager_workflow_id)

    try:
        return await client.start_workflow(
            SessionManagerWorkflow.run,
            registry,
            id=manager_workflow_id,
            task_queue=manager_task_queue,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
    except WorkflowAlreadyStartedError:
        return client.get_workflow_handle(manager_workflow_id)


# ---------------------------------------------------------------------------
# Message-shape introspection
# ---------------------------------------------------------------------------


def _pick_handler(interface: list[AcceptedFunction], message_type: str | None) -> AcceptedFunction:
    if message_type is not None:
        for fn in interface:
            if fn.name == message_type:
                return fn
        known = ", ".join(fn.name for fn in interface)
        raise SystemExit(f"No {message_type!r} handler. Known handlers: {known}")

    for fn in interface:
        if fn.name == "ask":
            return fn
    if len(interface) == 1:
        return interface[0]
    known = ", ".join(fn.name for fn in interface)
    raise SystemExit(
        f"Agent exposes multiple handlers ({known}) and none is named 'ask'; "
        "pick one with --message-type."
    )


def _infer_text_field(parameters: dict[str, Any]) -> str | None:
    """Best-effort: which property of this handler's input schema is "the free-text
    message"? Prefers a field literally named ``text`` (the harness's own ``TextMessage``
    convention); otherwise the sole required string field, or the sole string field."""
    properties: dict[str, Any] = parameters.get("properties") or {}
    string_fields = [name for name, schema in properties.items() if schema.get("type") == "string"]
    if "text" in string_fields:
        return "text"
    required = set(parameters.get("required") or [])
    required_strings = [name for name in string_fields if name in required]
    if len(required_strings) == 1:
        return required_strings[0]
    if len(string_fields) == 1:
        return string_fields[0]
    return None


# ---------------------------------------------------------------------------
# Turn streaming
# ---------------------------------------------------------------------------


async def _run_turn(
    agent_client: AgentClient,
    msg_type: str,
    payload: dict[str, Any],
    expected_turn: int,
) -> tuple[dict[str, Any] | None, bool]:
    """Send one turn and print its streamed events, prompting interactively for any
    tool-approval request. Returns ``(reply_output, ok)``."""
    reply_output: dict[str, Any] | None = None
    saw_delta = False
    ok = True

    stream = await agent_client.send_message(
        msg_type, payload, expected_turn, on_item=lambda ev, _resume_offset: ev
    )
    async for item in stream:
        if isinstance(item, AgentTurnTimeout):
            print(f"\n[timeout] {item}")
            ok = False
            continue
        if isinstance(item, AgentTurnError):
            print(f"\n[error] {item}")
            ok = False
            continue

        assert isinstance(item, AgentEvent)
        event = item.event
        if event.type == AgentEventType.TOOL_APPROVAL_REQUESTED:
            print(f"\n[approval needed] {event.tool_name}({json.dumps(event.tool_input)})")
            answer = input("Approve? [y/N] ").strip().lower()
            await agent_client.approve_tool(event.tool_id, approved=answer == "y")
        elif event.type == AgentEventType.REPLY_DELTA:
            saw_delta = True
            print(event.text, end="", flush=True)
        elif event.type == AgentEventType.TOOL_START:
            print(f"\n[tool] {event.tool_name}({json.dumps(event.tool_input)})")
        elif event.type == AgentEventType.TOOL_END:
            print(f"       -> {event.tool_output[:300]}")
        elif event.type == AgentEventType.TOOL_ERROR:
            print(f"       ! {event.message}")
        elif event.type == AgentEventType.REPLY:
            reply_output = event.output
        elif event.type == AgentEventType.ERROR:
            print(f"\n[error] {event.message}")
            ok = False

    if reply_output is not None and not saw_delta:
        text = reply_output.get("text")
        print(text if isinstance(text, str) else json.dumps(reply_output))
    else:
        print()
    return reply_output, ok


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


async def cmd_agents(args: argparse.Namespace) -> None:
    client = await _connect(args.address, args.namespace)
    registry = load_agent_registry(args.registry)
    manager = await _ensure_session_manager(
        client,
        registry,
        manager_workflow_id=args.manager_workflow_id,
        manager_task_queue=args.manager_task_queue,
    )
    result: AgentRegistry = await manager.query(
        SessionManagerWorkflow.available_agents, result_type=AgentRegistry
    )
    for descriptor in result.agents:
        print(f"{descriptor.key:20} {descriptor.label:24} {descriptor.description}")


async def cmd_sessions(args: argparse.Namespace) -> None:
    client = await _connect(args.address, args.namespace)
    registry = load_agent_registry(args.registry)
    manager = await _ensure_session_manager(
        client,
        registry,
        manager_workflow_id=args.manager_workflow_id,
        manager_task_queue=args.manager_task_queue,
    )
    sessions: list[Session] = await manager.query(
        SessionManagerWorkflow.list_sessions, result_type=list[Session]
    )
    known_ids = {session.workflow_id for session in sessions}
    discovered = await discover_untracked_sessions(client, registry, known_ids)

    for session in sessions + discovered:
        status = await workflow_execution_status(client, session.workflow_id)
        status_name = status.name if status is not None else "NOT_FOUND"
        origin = "discovered" if session.is_discovered else "tracked"
        print(
            f"{session.workflow_id}  [{status_name:10}] "
            f"{session.agent_workflow_type:24} {session.label:12} ({origin})"
        )


async def cmd_chat(args: argparse.Namespace) -> None:
    client = await _connect(args.address, args.namespace)

    if args.session:
        workflow_id = args.session
    else:
        if not args.agent:
            raise SystemExit("chat needs --session (resume an existing one) or --agent (start new)")
        registry = load_agent_registry(args.registry)
        manager = await _ensure_session_manager(
            client,
            registry,
            manager_workflow_id=args.manager_workflow_id,
            manager_task_queue=args.manager_task_queue,
        )
        descriptor = registry.by_key(args.agent)
        if descriptor is None:
            known = ", ".join(a.key for a in registry.agents)
            raise SystemExit(f"Unknown agent key {args.agent!r}. Known: {known}")
        session: Session = await manager.execute_update(
            SessionManagerWorkflow.create_session,
            CreateSessionRequest(agent_workflow_type=descriptor.workflow_type, config=AgentConfig()),
            result_type=Session,
        )
        workflow_id = session.workflow_id
        print(f"Started session {workflow_id}")

    agent_client = AgentClient(client, workflow_id)
    interface = await agent_client.get_agent_interface()
    if not interface:
        raise SystemExit(f"{workflow_id} exposes no @agent.accepts handlers.")
    handler = _pick_handler(interface, args.message_type)
    field = _infer_text_field(handler.parameters)
    if field is None:
        raise SystemExit(
            f"Can't infer a free-text field for '{handler.name}' from its schema "
            f"({json.dumps(handler.parameters)}). Pass --message-type to pick a different "
            "handler, or prefix a line with '/json ' to send a raw JSON payload."
        )

    status = await agent_client.get_status()
    expected_turn = status.current_turn + 1

    print(f"Chatting with {workflow_id} via '{handler.name}' (field {field!r}). Ctrl-D or /exit to quit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break

        if line.startswith("/json "):
            payload = json.loads(line[len("/json ") :])
        else:
            payload = {field: line}

        try:
            await _run_turn(agent_client, handler.name, payload, expected_turn)
        except StaleTurnError:
            status = await agent_client.get_status()
            expected_turn = status.current_turn + 1
            print(f"[stale turn — resynced to turn {expected_turn}; resend your message]")
            continue
        except AgentBusyError as exc:
            print(f"[busy] {exc}")
            continue
        expected_turn += 1


async def cmd_approve(args: argparse.Namespace) -> None:
    client = await _connect(args.address, args.namespace)
    agent_client = AgentClient(client, args.session)

    if args.tool_id is None:
        pending = await agent_client.get_pending_approvals()
        if not pending:
            print("No pending approvals.")
            return
        for approval in pending:
            print(
                f"{approval.tool_id}  {approval.tool_name}({json.dumps(approval.tool_input)})"
                f"  [turn {approval.turn_number}]"
            )
        return

    result = await agent_client.approve_tool(
        args.tool_id, approved=not args.deny, reason=args.reason, remember=args.remember
    )
    print(f"{'denied' if args.deny else 'approved'}: {result.tool_id}")


async def cmd_close(args: argparse.Namespace) -> None:
    client = await _connect(args.address, args.namespace)
    handle = client.get_workflow_handle(args.session)
    await handle.signal("close")
    print(f"Sent close signal to {args.session}")


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__.splitlines()[0])
    parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    parser.add_argument("--namespace", default=os.environ.get("TEMPORAL_NAMESPACE", "default"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_registry_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--registry", default="agents.toml", help="Agent registry TOML path")
        sub.add_argument("--manager-workflow-id", default=SESSION_MANAGER_ID)
        sub.add_argument("--manager-task-queue", default=SESSION_MANAGER_TASK_QUEUE)

    agents_parser = subparsers.add_parser("agents", help="List agents in the registry")
    add_registry_args(agents_parser)
    agents_parser.set_defaults(func=cmd_agents)

    sessions_parser = subparsers.add_parser("sessions", help="List known + discovered sessions")
    add_registry_args(sessions_parser)
    sessions_parser.set_defaults(func=cmd_sessions)

    chat_parser = subparsers.add_parser("chat", help="Start a new session or resume an existing one")
    add_registry_args(chat_parser)
    chat_parser.add_argument("--agent", help="Registry key of the agent to start a new session for")
    chat_parser.add_argument("--session", help="Workflow id of an existing session to resume")
    chat_parser.add_argument(
        "--message-type",
        help="Name of the @agent.accepts handler to use (default: 'ask', or the only one)",
    )
    chat_parser.set_defaults(func=cmd_chat)

    approve_parser = subparsers.add_parser(
        "approve", help="List, or resolve, an agent's pending tool approvals"
    )
    approve_parser.add_argument("--session", required=True, help="Workflow id of the session")
    approve_parser.add_argument("--tool-id", help="Approve/deny this pending call (default: list)")
    approve_parser.add_argument("--deny", action="store_true", help="Deny instead of approve")
    approve_parser.add_argument("--reason", help="Note attached to the decision")
    approve_parser.add_argument(
        "--remember", action="store_true", help="Also auto-approve future calls of this tool"
    )
    approve_parser.set_defaults(func=cmd_approve)

    close_parser = subparsers.add_parser("close", help="Gracefully stop a session")
    close_parser.add_argument("--session", required=True, help="Workflow id of the session")
    close_parser.set_defaults(func=cmd_close)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
