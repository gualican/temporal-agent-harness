"""Opt-in "auto" model routing for the OpenAI Agents integration, using jev-model-router's
Jev-based classifier to pick a model per call instead of one fixed model name.

Why this lives here, not in a custom ModelProvider: ``ModelProvider.get_model(model_name)``
receives only the model name string, never the conversation, so it cannot make a
content-based routing decision. Routing has to happen one layer up, inside the Temporal
*activity* that resolves the model - which is also exactly where an extra API call belongs,
since activities are Temporal's designated place for non-deterministic IO.

``OpenAIAgentsPlugin`` has no way to inject a custom ``ModelActivity`` instance - it always
constructs the base class itself when ``register_activities=True`` (the default). So use
this with ``register_activities=False`` on the plugin, and register this activity's two
methods on the ``Worker`` yourself::

    plugin = OpenAIAgentsPlugin(model_params=..., register_activities=False)
    jev_activity = JevAutoModelActivity(
        tier_to_model={"simple": "gpt-5-mini", "moderate": "gpt-5.1", "complex": "gpt-5.1"},
        default_model="gpt-5.1",
    )
    worker = Worker(
        client,
        activities=[
            jev_activity.invoke_model_activity,
            jev_activity.invoke_model_activity_streaming,
        ],
        ...,
    )

Then set ``Agent(model=AUTO_MODEL, ...)`` on any agent that should route automatically -
agents with an explicit model name are unaffected; this activity behaves exactly like the
base ``ModelActivity`` for every ``model_name`` other than ``AUTO_MODEL``.

Known limitation: a single user turn can trigger several ``invoke_model_activity`` calls as
the agent works through tool calls. This resolves the routing decision independently on
each one (the extracted user text is unchanged within a turn, so the result is almost
always identical) rather than caching it across those calls - an acceptable v1
simplification given Jev's low cost/latency; caching would need workflow-level state.
"""

import logging

from agents import ModelResponse
from agents.items import TResponseStreamEvent
from jev_model_router.classifier import classify
from jev_model_router.router import resolve_tier
from typesafe_sdk import TypeSafeClient

from temporalio import activity

from temporal_agent_harness.ai_sdks.openai_agents._heartbeat_decorator import auto_heartbeater
from temporal_agent_harness.ai_sdks.openai_agents._invoke_model_activity import (
    ActivityModelInput,
    ModelActivity,
    StreamingActivityModelInput,
)

AUTO_MODEL = "auto"

_logger = logging.getLogger(__name__)


def _extract_latest_user_text(items: list) -> str | None:
    """Find the most recent user message's text in an SDK input-item list.

    Scans from the end rather than trusting ``items[-1]``: mid-tool-loop activity
    invocations can end in a ``function_call_output`` or similar non-message item, not a
    fresh user message.
    """
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        if item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                part.get("text")
                for part in content
                if isinstance(part, dict) and part.get("type") == "input_text" and part.get("text")
            ]
            if texts:
                return "\n".join(texts)
    return None


class JevAutoModelActivity(ModelActivity):
    """A ``ModelActivity`` that resolves ``AUTO_MODEL`` requests via jev-model-router's
    classifier, and behaves exactly like the base ``ModelActivity`` for every other
    ``model_name``.
    """

    def __init__(
        self,
        tier_to_model: dict[str, str],
        *,
        default_model: str,
        model_provider=None,
        observer_factory=None,
    ):
        super().__init__(model_provider, observer_factory=observer_factory)
        self._tier_to_model = tier_to_model
        self._default_model = default_model
        self._typesafe_client: TypeSafeClient | None = None

    def _get_typesafe_client(self) -> TypeSafeClient:
        # Lazy, not eager: constructing in __init__ would require TYPESAFE_API_KEY even for
        # callers/tests that never route through AUTO_MODEL. No lock needed - this activity
        # is async, so Temporal runs it as an asyncio task on the worker's event loop, not a
        # thread pool; with no `await` between the check and the assignment, no other
        # coroutine can interleave mid-check the way concurrent OS threads could.
        if self._typesafe_client is None:
            self._typesafe_client = TypeSafeClient()
        return self._typesafe_client

    @activity.defn(name="invoke_model_activity")
    @auto_heartbeater
    async def invoke_model_activity(self, input: ActivityModelInput) -> ModelResponse:
        return await super().invoke_model_activity(await self._resolve_auto(input))

    @activity.defn(name="invoke_model_activity_streaming")
    @auto_heartbeater
    async def invoke_model_activity_streaming(
        self, input: StreamingActivityModelInput
    ) -> list[TResponseStreamEvent]:
        return await super().invoke_model_activity_streaming(await self._resolve_auto(input))

    async def _resolve_auto(self, input: ActivityModelInput) -> ActivityModelInput:
        if input.get("model_name") != AUTO_MODEL:
            return input

        raw_input = input["input"]
        text = raw_input if isinstance(raw_input, str) else _extract_latest_user_text(raw_input)
        if not text:
            _logger.info("jev auto-routing: no user text found, using default_model=%s", self._default_model)
            return {**input, "model_name": self._default_model}

        classification = classify(text, client=self._get_typesafe_client())
        tier, fallback, override = resolve_tier(classification)
        resolved_model = self._tier_to_model.get(tier, self._default_model)
        _logger.info(
            "jev auto-routing: tier=%s (confidence=%.2f, fallback=%s, stakes_override=%s) -> model=%s",
            tier, classification.tier_confidence, fallback, override, resolved_model,
        )
        return {**input, "model_name": resolved_model}
