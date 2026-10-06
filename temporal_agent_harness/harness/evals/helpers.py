# ABOUTME: Workflow-side helper that attaches an eval to a turn: ONLINE blocks on a child
# EvalWorkflow and returns the report; SIDE_EFFECT starts it abandoned (fire-and-forget) and
# returns its workflow id. Call it from inside an agent's @agent.accepts handler.

from __future__ import annotations

from temporalio import workflow
from temporalio.workflow import ParentClosePolicy

from temporal_agent_harness.harness.evals.models import (
    DEFAULT_JUDGE_MODEL,
    AgentRunResult,
    EvalMode,
    EvalReport,
    EvalRequest,
    EvalSpec,
)
from temporal_agent_harness.harness.evals.workflow import EvalWorkflow


async def run_turn_eval(
    run: AgentRunResult,
    spec: EvalSpec,
    mode: EvalMode | str,
    *,
    agent_instructions: str = "",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[EvalReport | None, str | None]:
    """Returns ``(report, None)`` for ONLINE, ``(None, eval_workflow_id)`` for SIDE_EFFECT, and
    ``(None, None)`` for any other mode (nothing is run)."""
    mode = EvalMode(mode)
    if mode not in (EvalMode.ONLINE, EvalMode.SIDE_EFFECT):
        return None, None
    request = EvalRequest(
        run=run,
        trigger=mode,
        spec=spec,
        agent_instructions=agent_instructions,
        judge_model=judge_model,
    )
    eval_id = f"{workflow.info().workflow_id}-eval-{workflow.uuid4().hex[:8]}"
    if mode == EvalMode.ONLINE:
        report = await workflow.execute_child_workflow(EvalWorkflow.run, request, id=eval_id)
        return report, None
    handle = await workflow.start_child_workflow(
        EvalWorkflow.run,
        request,
        id=eval_id,
        parent_close_policy=ParentClosePolicy.ABANDON,
    )
    return None, handle.id
