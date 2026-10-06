# ABOUTME: Payload models for evals: the spec an agent is graded against, the run being graded, the
# request to grade it, and the report. All inherit the shared ``Model`` (see harness.payload_model).

from __future__ import annotations

from enum import Enum

from pydantic import Field

from temporal_agent_harness.harness.payload_model import Model

DEFAULT_JUDGE_MODEL = "gpt-5-mini"


class EvalMode(str, Enum):
    """How (and whether) an eval is attached to an agent run."""

    NONE = "none"
    # Eval runs as a blocking child workflow; the agent run returns the report inline.
    ONLINE = "online"
    # Eval runs as an abandoned (fire-and-forget) child workflow after the agent run.
    SIDE_EFFECT = "side_effect"
    # Not set on agent runs; used as the trigger label when evals are run standalone.
    OFFLINE = "offline"


class EvalSpec(Model):
    """How runs of one agent are graded."""

    # Criteria scored 1-5 by the LLM judge.
    judge_criteria: list[str]
    # Judge passes when the mean criterion score is >= this threshold.
    judge_pass_threshold: float = 3.5
    # Programmatic checks.
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    output_must_contain: list[str] = []
    output_must_not_contain: list[str] = []
    max_tool_calls: int | None = None


class ToolCallRecord(Model):
    name: str
    arguments: str = ""
    output: str = ""


class CheckResult(Model):
    name: str
    passed: bool
    details: str = ""


class CriterionScore(Model):
    criterion: str
    score: int = Field(ge=1, le=5, description="1 = fails the criterion, 5 = fully satisfies it")
    reasoning: str


class JudgeVerdict(Model):
    criterion_scores: list[CriterionScore]
    summary: str

    @property
    def mean_score(self) -> float:
        if not self.criterion_scores:
            return 0.0
        return sum(c.score for c in self.criterion_scores) / len(self.criterion_scores)


class EvalReport(Model):
    agent_name: str
    agent_workflow_id: str
    trigger: EvalMode
    checks: list[CheckResult]
    checks_passed: bool
    judge: JudgeVerdict | None = None
    judge_mean_score: float = 0.0
    passed: bool


class AgentRunResult(Model):
    """One agent turn, as the grader sees it."""

    agent_name: str
    prompt: str
    final_output: str
    tool_calls: list[ToolCallRecord] = []
    workflow_id: str
    # Populated only for ONLINE eval mode.
    eval_report: EvalReport | None = None
    # Populated only for SIDE_EFFECT eval mode.
    eval_workflow_id: str | None = None


class EvalRequest(Model):
    run: AgentRunResult
    trigger: EvalMode
    # What to grade against. Carried in the request (not looked up) so the workflow is agnostic
    # to where agent definitions live.
    spec: EvalSpec
    # The agent's system prompt, shown to the judge as context for what the agent was told to do.
    agent_instructions: str = ""
    judge_model: str = DEFAULT_JUDGE_MODEL
