"""A minimal workflow used only by the eval tests: attaches an eval to a canned turn through
``run_turn_eval`` so ONLINE / SIDE_EFFECT behavior is tested inside a real workflow. Kept in its own
module because the workflow sandbox re-imports a workflow's defining module.

No ``from __future__ import annotations`` (types cross the data converter)."""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_agent_harness.harness.evals.helpers import run_turn_eval
    from temporal_agent_harness.harness.evals.models import (
        AgentRunResult,
        EvalReport,
        EvalSpec,
    )
    from temporal_agent_harness.harness.payload_model import Model


class TurnEvalInput(Model):
    mode: str
    spec: EvalSpec


class TurnEvalOutput(Model):
    report: EvalReport | None = None
    eval_workflow_id: str | None = None


@workflow.defn(name="TurnEvalHelper")
class TurnEvalHelperWorkflow:
    @workflow.run
    async def run(self, request: TurnEvalInput) -> TurnEvalOutput:
        run = AgentRunResult(
            agent_name="helper-agent",
            prompt="What is 23 * 19?",
            final_output="437",
            workflow_id=workflow.info().workflow_id,
        )
        report, eval_id = await run_turn_eval(
            run, request.spec, request.mode, agent_instructions="Be a calculator."
        )
        return TurnEvalOutput(report=report, eval_workflow_id=eval_id)
