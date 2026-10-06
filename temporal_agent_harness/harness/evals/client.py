# ABOUTME: Client-side eval helpers: grade an already-finished run offline, and render a report.
# Not for workflow code (holds a Temporal client). Gathering the runs to grade — past sessions,
# a dataset — is agent-specific, so it stays with the caller.

import uuid

from temporalio.client import Client

from temporal_agent_harness.harness.evals.models import (
    DEFAULT_JUDGE_MODEL,
    AgentRunResult,
    EvalMode,
    EvalReport,
    EvalRequest,
    EvalSpec,
)
from temporal_agent_harness.harness.evals.workflow import EvalWorkflow


async def evaluate_run(
    client: Client,
    run: AgentRunResult,
    spec: EvalSpec,
    *,
    task_queue: str,
    agent_instructions: str = "",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> EvalReport:
    """Grade ``run`` with an OFFLINE ``EvalWorkflow`` started on ``task_queue``."""
    return await client.execute_workflow(
        EvalWorkflow.run,
        EvalRequest(
            run=run,
            trigger=EvalMode.OFFLINE,
            spec=spec,
            agent_instructions=agent_instructions,
            judge_model=judge_model,
        ),
        id=f"eval-offline-{run.workflow_id}-{uuid.uuid4().hex[:8]}",
        task_queue=task_queue,
    )


def format_report(report: EvalReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [f"[{status}] {report.agent_name} run={report.agent_workflow_id}"]
    for check in report.checks:
        mark = "ok" if check.passed else "FAIL"
        lines.append(f"  check {mark}: {check.name} {check.details}".rstrip())
    if report.judge:
        for cs in report.judge.criterion_scores:
            lines.append(f"  judge {cs.score}/5: {cs.criterion} — {cs.reasoning}")
        lines.append(f"  judge mean: {report.judge_mean_score:.2f}")
    return "\n".join(lines)
