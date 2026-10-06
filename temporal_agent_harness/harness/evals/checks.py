# ABOUTME: Deterministic (programmatic) eval checks. Pure functions, safe in workflow code.

from __future__ import annotations

from temporal_agent_harness.harness.evals.models import AgentRunResult, CheckResult, EvalSpec


def run_checks(run: AgentRunResult, spec: EvalSpec) -> list[CheckResult]:
    results: list[CheckResult] = []
    called = [t.name for t in run.tool_calls]
    output = run.final_output or ""

    for tool in spec.required_tools:
        results.append(
            CheckResult(
                name=f"required_tool:{tool}",
                passed=tool in called,
                details=f"tools called: {called}",
            )
        )
    for tool in spec.forbidden_tools:
        results.append(
            CheckResult(
                name=f"forbidden_tool:{tool}",
                passed=tool not in called,
                details=f"tools called: {called}",
            )
        )
    for needle in spec.output_must_contain:
        results.append(
            CheckResult(
                name=f"output_contains:{needle}",
                passed=needle.lower() in output.lower(),
            )
        )
    for needle in spec.output_must_not_contain:
        results.append(
            CheckResult(
                name=f"output_excludes:{needle}",
                passed=needle.lower() not in output.lower(),
            )
        )
    if spec.max_tool_calls is not None:
        results.append(
            CheckResult(
                name=f"max_tool_calls:{spec.max_tool_calls}",
                passed=len(run.tool_calls) <= spec.max_tool_calls,
                details=f"{len(run.tool_calls)} tool calls",
            )
        )
    results.append(
        CheckResult(
            name="non_empty_output",
            passed=bool(output.strip()),
        )
    )
    return results
