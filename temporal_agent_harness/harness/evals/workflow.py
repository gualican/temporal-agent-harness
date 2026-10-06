# ABOUTME: EvalWorkflow grades a single agent run with programmatic checks + an LLM judge.
#
# The same workflow serves all three invocation modes; only the trigger differs:
# - online: executed as a blocking child of the agent workflow
# - side_effect: started as an ABANDON child of the agent workflow
# - offline: started directly by a client over past runs or a dataset
# The judge runs through the OpenAI Agents SDK, so the worker's client needs the OpenAI Agents plugin.

from __future__ import annotations

from temporalio import workflow

from agents import Agent, Runner

from temporal_agent_harness.harness.evals.checks import run_checks
from temporal_agent_harness.harness.evals.models import EvalReport, EvalRequest, JudgeVerdict

JUDGE_INSTRUCTIONS = """\
You are an impartial evaluator of AI agent runs. You will receive a transcript of
an agent run (the agent's system prompt, the user prompt, the tool calls it made,
and its final answer) followed by a list of evaluation criteria.

Score each criterion from 1 (clearly fails) to 5 (fully satisfies), with short,
specific reasoning grounded in the transcript. Judge only what is in the
transcript; do not reward claims the transcript does not support."""


def _format_transcript(request: EvalRequest) -> str:
    run = request.run
    instructions = request.agent_instructions or "(not provided)"
    lines = [
        "## Agent system prompt",
        instructions,
        "",
        "## User prompt",
        run.prompt,
        "",
        "## Tool calls",
    ]
    if run.tool_calls:
        for call in run.tool_calls:
            lines.append(f"- {call.name}({call.arguments}) -> {call.output}")
    else:
        lines.append("(none)")
    lines += ["", "## Final answer", run.final_output, "", "## Criteria to score"]
    lines += [f"{i + 1}. {c}" for i, c in enumerate(request.spec.judge_criteria)]
    return "\n".join(lines)


@workflow.defn
class EvalWorkflow:
    @workflow.run
    async def run(self, request: EvalRequest) -> EvalReport:
        spec = request.spec

        checks = run_checks(request.run, spec)
        checks_passed = all(c.passed for c in checks)

        judge_agent = Agent(
            name="eval-judge",
            instructions=JUDGE_INSTRUCTIONS,
            model=request.judge_model,
            output_type=JudgeVerdict,
        )
        judge_result = await Runner.run(
            judge_agent,
            _format_transcript(request),
        )
        verdict: JudgeVerdict = judge_result.final_output
        judge_passed = verdict.mean_score >= spec.judge_pass_threshold

        return EvalReport(
            agent_name=request.run.agent_name,
            agent_workflow_id=request.run.workflow_id,
            trigger=request.trigger,
            checks=checks,
            checks_passed=checks_passed,
            judge=verdict,
            judge_mean_score=verdict.mean_score,
            passed=checks_passed and judge_passed,
        )
