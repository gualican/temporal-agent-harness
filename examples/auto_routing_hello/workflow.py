"""A hello-world OpenAI Agents SDK agent demonstrating jev-model-router's auto model routing.

Identical to examples/openai_hello, except the agent's model starts as AUTO_MODEL: each
model call is classified by Jev (via JevAutoModelActivity, wired in worker.py) and routed to
TIER_TO_MODEL[tier] instead of always calling one fixed model. A /model slash command lets an
operator switch a running session between auto and any single model in TIER_TO_MODEL, the
same way examples/monty's conversational_workflow does for its own fixed model list.

Run it with the shared example stack (session-manager worker + FastAPI/UI); this agent is
registered in agents.toml and driven by the packaged web app. See README.md.
"""

from __future__ import annotations

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream

with workflow.unsafe.imports_passed_through():
    from agents import Agent as OpenAIAgent
    from agents import Runner, TResponseInputItem

    from temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing import AUTO_MODEL
    from temporal_agent_harness.ai_sdks.openai_agents_harness import as_openai_agent_tool
    from temporal_agent_harness.harness import agent, slash_commands
    from temporal_agent_harness.harness.agent_protocol import (
        AgentConfig,
        TextMessage,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner


TASK_QUEUE = "auto-routing-hello"

# jev_model_router's own TIER_TO_MODEL routes Claude models; this harness is multi-provider,
# so the app picks its own three model names for the tiers Jev classifies into.
TIER_TO_MODEL = {
    "simple": "gpt-5-mini",
    "moderate": "gpt-5.1",
    "complex": "gpt-5.1",
}

SYSTEM_INSTRUCTION = """\
You are a friendly assistant. Answer the user in brief, natural prose.

You have one tool, `get_weather`, which returns the current weather for a city. When the user
asks about the weather somewhere, call it (don't guess), then tell them the answer in a sentence
or two. For anything else, just reply directly."""


@agent.tool_defn(inherently_safe=True)
async def get_weather(city: str) -> str:
    """Return the current weather for a city. `city` is a plain city name, e.g. "Paris"."""
    # Canned lookup — a hello-world, not a real weather service.
    return f"It's 72°F and sunny in {city}."


@workflow.defn(name="AutoRoutingHelloAgent")
@agent.defn
class AutoRoutingHelloAgentWorkflow:
    """A one-tool conversational agent, routed per-call by jev-model-router instead of a
    single fixed model."""

    @workflow.init
    def __init__(self, config: AgentConfig) -> None:
        self._model = AUTO_MODEL

        def _set_model(model: str) -> None:
            self._model = model

        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            # Hello-world stance: don't gate tool calls. A caller can tighten this per
            # session via AgentConfig.approval_policy.
            approval_policy_default=ToolApprovalPolicy.dangerously_skip_all(),
            slash_commands=[
                *slash_commands.default_commands(),
                slash_commands.model_selector(
                    # dict.fromkeys(...) dedupes while preserving order — TIER_TO_MODEL maps
                    # more than one tier to the same model name below.
                    choices=(AUTO_MODEL, *dict.fromkeys(TIER_TO_MODEL.values())),
                    set_model=_set_model,
                    description="Set the model for this session (or 'auto' to route per-call via Jev).",
                ),
            ],
        )
        # OpenAI conversation state, threaded across turns as the SDK's input-item list.
        self._conversation: list[TResponseInputItem] = []

    @workflow.run
    async def run(self, _config: AgentConfig) -> None:
        await self._runner.run(self)

    @agent.accepts
    async def ask(self, message: TextMessage) -> TextReply:
        """Chat with the assistant. Ask it anything; ask about the weather in a city and it
        calls its `get_weather` tool and tells you what it found. Use `/model` to switch
        between 'auto' (Jev routes per-call) and a fixed model."""
        sdk_agent = OpenAIAgent(
            name="AutoRoutingHello",
            instructions=SYSTEM_INSTRUCTION,
            model=self._model,
            tools=[as_openai_agent_tool(self._runner, get_weather)],
        )
        input_items: list[TResponseInputItem] = [
            *self._conversation,
            {"role": "user", "content": message.text},
        ]

        result = Runner.run_streamed(sdk_agent, input=input_items, context=self._runner)
        async for _event in result.stream_events():
            pass

        self._conversation = result.to_input_list()
        return TextReply(text=str(result.final_output))
