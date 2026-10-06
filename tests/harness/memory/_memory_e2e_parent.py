"""A minimal, MODEL-FREE parent agent used only by the memory tools' end-to-end test.

Its single handler drives the example's real memory tools through ``run_tool`` with the same
``user_id`` injection the example workflow uses, so a ``WorkflowEnvironment`` test exercises the
whole tool path (dispatcher -> activity -> MemoryProvider) without a model or a Mem0 account.

Kept in its own module because the workflow sandbox re-imports a workflow's defining module.
No ``from __future__ import annotations`` (activity/handler types cross the data converter).
"""

from temporalio.contrib.workflow_streams import WorkflowStream
from pydantic import BaseModel
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import (
        AgentConfig,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner
    from temporal_agent_harness.harness.memory import (
        forget_memory,
        list_memories,
        recall_memories,
        remember_memory,
    )


class Step(BaseModel):
    """One memory tool call: which tool, for which user, with what argument."""

    tool: str
    user_id: str
    arg: str = ""


@workflow.defn(name="MemoryE2EParent")
@agent.defn
class MemoryE2EParentWorkflow:
    @workflow.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            approval_policy_default=ToolApprovalPolicy.dangerously_skip_all(),
        )
        self._n = 0

    @workflow.run
    async def run(self, _config: AgentConfig) -> None:
        await self._runner.run(self)

    @agent.accepts
    async def step(self, msg: Step) -> TextReply:
        """Run one memory tool as ``msg.user_id`` and reply with its text."""
        self._n += 1
        inj = {"scope": msg.user_id}
        call_id = f"call-{self._n}"
        if msg.tool == "remember":
            out = await self._runner.run_tool(call_id, remember_memory, text=msg.arg, injections=inj)
        elif msg.tool == "recall":
            out = await self._runner.run_tool(call_id, recall_memories, query=msg.arg, injections=inj)
        elif msg.tool == "list":
            out = await self._runner.run_tool(call_id, list_memories, injections=inj)
        else:
            out = await self._runner.run_tool(call_id, forget_memory, memory_id=msg.arg, injections=inj)
        return TextReply(text=out)
