# ABOUTME: Unit tests for the deterministic eval checks (pure functions, no Temporal needed).

from temporal_agent_harness.harness.evals.checks import run_checks
from temporal_agent_harness.harness.evals.models import AgentRunResult, EvalSpec, ToolCallRecord


def _run(output: str = "the answer is 437", tools: tuple[str, ...] = ("calculate",)) -> AgentRunResult:
    return AgentRunResult(
        agent_name="a",
        prompt="p",
        final_output=output,
        tool_calls=[ToolCallRecord(name=t) for t in tools],
        workflow_id="wf",
    )


def _by_name(results) -> dict[str, bool]:
    return {r.name: r.passed for r in results}


def test_required_and_forbidden_tools():
    spec = EvalSpec(
        judge_criteria=["x"], required_tools=["calculate", "search"], forbidden_tools=["rm"]
    )
    got = _by_name(run_checks(_run(), spec))
    assert got["required_tool:calculate"] is True
    assert got["required_tool:search"] is False
    assert got["forbidden_tool:rm"] is True
    assert _by_name(run_checks(_run(tools=("rm",)), spec))["forbidden_tool:rm"] is False


def test_output_contains_is_case_insensitive_and_excludes():
    spec = EvalSpec(
        judge_criteria=["x"], output_must_contain=["ANSWER"], output_must_not_contain=["sorry"]
    )
    got = _by_name(run_checks(_run(), spec))
    assert got["output_contains:ANSWER"] is True
    assert got["output_excludes:sorry"] is True
    assert _by_name(run_checks(_run("Sorry, no"), spec))["output_excludes:sorry"] is False


def test_max_tool_calls_and_non_empty_output():
    spec = EvalSpec(judge_criteria=["x"], max_tool_calls=1)
    assert _by_name(run_checks(_run(tools=("a", "b")), spec))["max_tool_calls:1"] is False
    assert _by_name(run_checks(_run(tools=("a",)), spec))["max_tool_calls:1"] is True
    assert _by_name(run_checks(_run(output="  "), spec))["non_empty_output"] is False
    assert _by_name(run_checks(_run(), spec))["non_empty_output"] is True
