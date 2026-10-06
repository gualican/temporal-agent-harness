# ABOUTME: End-to-end tests of EvalWorkflow and the run_turn_eval helper with a SCRIPTED judge (the
# harness's TestModelProvider stands in for OpenAI), proving the spec travels in the request, the
# pass/fail logic combines checks with the judge, and the three trigger modes behave.
#
# Run with: uv run pytest tests/harness/evals -v

import json
import uuid
from collections import deque
from datetime import timedelta

import pytest_asyncio
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_agent_harness.ai_sdks.openai_agents import (
    ModelActivityParameters,
    OpenAIAgentsPlugin,
)
from temporal_agent_harness.ai_sdks.openai_agents.testing import (
    ResponseBuilders,
    TestModel,
    TestModelProvider,
)
from temporal_agent_harness.harness.evals.client import evaluate_run, format_report
from temporal_agent_harness.harness.evals.models import (
    AgentRunResult,
    EvalMode,
    EvalReport,
    EvalSpec,
)
from temporal_agent_harness.harness.evals.workflow import EvalWorkflow

from ._eval_helper_workflow import TurnEvalHelperWorkflow, TurnEvalInput, TurnEvalOutput


def _judge_json(*scores: int) -> str:
    return json.dumps(
        {
            "criterion_scores": [
                {"criterion": f"c{i}", "score": s, "reasoning": "because"}
                for i, s in enumerate(scores)
            ],
            "summary": "ok",
        }
    )


@pytest_asyncio.fixture
async def stack():
    """Yields (client, task_queue, script) where ``script(*scores)`` queues one judge verdict."""
    responses: deque = deque()
    env = await WorkflowEnvironment.start_time_skipping()
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=30)),
        model_provider=TestModelProvider(TestModel(lambda: responses.popleft())),
    )
    config = env.client.config()
    config["plugins"] = [plugin]
    client = Client(**config)
    queue = f"evals-{uuid.uuid4()}"
    try:
        async with Worker(
            client, task_queue=queue, workflows=[EvalWorkflow, TurnEvalHelperWorkflow]
        ):
            yield client, queue, lambda *s: responses.append(
                ResponseBuilders.output_message(_judge_json(*s))
            )
    finally:
        await env.shutdown()


def _run() -> AgentRunResult:
    from temporal_agent_harness.harness.evals.models import ToolCallRecord

    return AgentRunResult(
        agent_name="calc",
        prompt="What is 23 * 19?",
        final_output="437",
        tool_calls=[ToolCallRecord(name="calculate", arguments="{}", output="437")],
        workflow_id="session-1",
    )


async def test_offline_passes_when_checks_and_judge_pass(stack):
    client, queue, judge = stack
    judge(5, 4)
    spec = EvalSpec(judge_criteria=["a", "b"], required_tools=["calculate"])
    report = await evaluate_run(client, _run(), spec, task_queue=queue)
    assert report.trigger == EvalMode.OFFLINE
    assert report.checks_passed and report.passed
    assert report.judge_mean_score == 4.5
    text = format_report(report)
    assert text.startswith("[PASS] calc") and "judge mean: 4.50" in text


async def test_low_judge_score_fails_the_run(stack):
    client, queue, judge = stack
    judge(2, 3)  # mean 2.5 < default threshold 3.5
    report = await evaluate_run(client, _run(), EvalSpec(judge_criteria=["a", "b"]), task_queue=queue)
    assert report.checks_passed and not report.passed


async def test_failed_check_fails_the_run_even_if_judge_is_happy(stack):
    client, queue, judge = stack
    judge(5, 5)
    spec = EvalSpec(judge_criteria=["a", "b"], required_tools=["search"])  # never called
    report = await evaluate_run(client, _run(), spec, task_queue=queue)
    assert not report.checks_passed and not report.passed
    assert format_report(report).startswith("[FAIL]")


async def test_threshold_comes_from_the_request_spec(stack):
    client, queue, judge = stack
    judge(2, 3)
    spec = EvalSpec(judge_criteria=["a", "b"], judge_pass_threshold=2.0)
    assert (await evaluate_run(client, _run(), spec, task_queue=queue)).passed


async def test_turn_helper_online_returns_report(stack):
    client, queue, judge = stack
    judge(5)
    out = await client.execute_workflow(
        TurnEvalHelperWorkflow.run,
        TurnEvalInput(mode="online", spec=EvalSpec(judge_criteria=["a"])),
        id=f"helper-{uuid.uuid4()}",
        task_queue=queue,
    )
    assert out.eval_workflow_id is None
    assert out.report is not None and out.report.trigger == EvalMode.ONLINE and out.report.passed


async def test_turn_helper_side_effect_returns_workflow_id(stack):
    client, queue, judge = stack
    judge(5)
    out = await client.execute_workflow(
        TurnEvalHelperWorkflow.run,
        TurnEvalInput(mode="side_effect", spec=EvalSpec(judge_criteria=["a"])),
        id=f"helper-{uuid.uuid4()}",
        task_queue=queue,
    )
    assert out.report is None and out.eval_workflow_id
    report = await client.get_workflow_handle(
        out.eval_workflow_id, result_type=EvalReport
    ).result()
    assert report.trigger == EvalMode.SIDE_EFFECT and report.passed


async def test_turn_helper_none_mode_runs_nothing(stack):
    client, queue, _judge = stack  # no verdict queued: a judge call would raise IndexError
    out = await client.execute_workflow(
        TurnEvalHelperWorkflow.run,
        TurnEvalInput(mode="none", spec=EvalSpec(judge_criteria=["a"])),
        id=f"helper-{uuid.uuid4()}",
        task_queue=queue,
    )
    assert out == TurnEvalOutput()
