# ABOUTME: Unit-tests JevAutoModelActivity's routing logic in isolation — the user-text
# extraction helper against various SDK input-item shapes, the AUTO_MODEL resolution path
# with jev_model_router.classifier.classify mocked (no real API calls, no Temporal server),
# and one end-to-end pass through the real @activity.defn-decorated method via
# temporalio.testing.ActivityEnvironment to confirm the ModelActivity subclass actually
# registers and runs correctly.
#
# Run with: uv run pytest tests/ai_sdks/openai_agents/test_jev_auto_routing.py -v

from __future__ import annotations

from unittest.mock import patch

import pytest
from agents import ModelSettings

from temporalio.testing import ActivityEnvironment

from jev_model_router.classifier import Classification

from temporal_agent_harness.ai_sdks.openai_agents._invoke_model_activity import (
    ModelTracingInput,
)
from temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing import (
    AUTO_MODEL,
    JevAutoModelActivity,
    _extract_latest_user_text,
)
from temporal_agent_harness.ai_sdks.openai_agents.testing import (
    ResponseBuilders,
    TestModel,
    TestModelProvider,
)

TIER_TO_MODEL = {"simple": "small-model", "moderate": "mid-model", "complex": "big-model"}
DEFAULT_MODEL = "mid-model"


@pytest.fixture(autouse=True)
def _typesafe_api_key(monkeypatch):
    # JevAutoModelActivity constructs a real TypeSafeClient on first use (see
    # _get_typesafe_client); classify() itself is always mocked below, so the client is
    # never used for a real call, but its constructor still requires a key to be present.
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def _classification(tier: str, tier_confidence: float = 0.9, stakes_score: float = 0.0) -> Classification:
    return Classification(
        tier=tier, tier_confidence=tier_confidence, stakes_score=stakes_score, stakes_confidence=0.9
    )


def _base_input(model_name: str | None, input_items) -> dict:
    return {
        "model_name": model_name,
        "input": input_items,
        "model_settings": ModelSettings(),
        "tracing": ModelTracingInput.DISABLED,
    }


class TestExtractLatestUserText:
    def test_single_first_turn_user_message_str_content(self):
        items = [{"role": "user", "content": "hello there"}]
        assert _extract_latest_user_text(items) == "hello there"

    def test_list_of_parts_content(self):
        items = [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "part one"},
                    {"type": "input_text", "text": "part two"},
                ],
            }
        ]
        assert _extract_latest_user_text(items) == "part one\npart two"

    def test_multi_turn_history_ending_in_user_message(self):
        items = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "first answer"}]},
            {"role": "user", "content": "second question"},
        ]
        assert _extract_latest_user_text(items) == "second question"

    def test_skips_trailing_non_user_tool_item(self):
        items = [
            {"role": "user", "content": "book me a flight"},
            {"type": "function_call", "call_id": "1", "name": "search_flights", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "1", "output": "3 flights found"},
        ]
        assert _extract_latest_user_text(items) == "book me a flight"

    def test_empty_list_returns_none(self):
        assert _extract_latest_user_text([]) is None

    def test_no_user_item_returns_none(self):
        items = [{"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}]
        assert _extract_latest_user_text(items) is None


class TestResolveAuto:
    def _activity(self) -> JevAutoModelActivity:
        return JevAutoModelActivity(
            tier_to_model=TIER_TO_MODEL,
            default_model=DEFAULT_MODEL,
            model_provider=TestModelProvider(TestModel.returning_responses([ResponseBuilders.output_message("ok")])),
        )

    @pytest.mark.asyncio
    async def test_non_auto_model_name_passes_through_untouched(self):
        activity = self._activity()
        input = _base_input("explicit-model", [{"role": "user", "content": "hi"}])

        with patch("temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing.classify") as mock_classify:
            resolved = await activity._resolve_auto(input)

        mock_classify.assert_not_called()
        assert resolved is input

    @pytest.mark.asyncio
    async def test_auto_resolves_via_tier_to_model(self):
        activity = self._activity()
        input = _base_input(AUTO_MODEL, [{"role": "user", "content": "a genuinely hard question"}])

        with patch(
            "temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing.classify",
            return_value=_classification("complex"),
        ) as mock_classify:
            resolved = await activity._resolve_auto(input)

        mock_classify.assert_called_once()
        assert mock_classify.call_args.args[0] == "a genuinely hard question"
        assert resolved["model_name"] == "big-model"

    @pytest.mark.asyncio
    async def test_auto_with_no_extractable_user_text_falls_back_to_default(self):
        activity = self._activity()
        input = _base_input(AUTO_MODEL, [{"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}])

        with patch("temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing.classify") as mock_classify:
            resolved = await activity._resolve_auto(input)

        mock_classify.assert_not_called()
        assert resolved["model_name"] == DEFAULT_MODEL


@pytest.mark.asyncio
async def test_invoke_model_activity_end_to_end_via_activity_environment():
    """Confirms the subclassed, re-decorated @activity.defn method actually runs (not just
    _resolve_auto in isolation) and that AUTO_MODEL correctly picks the routed TestModel."""
    activity = JevAutoModelActivity(
        tier_to_model=TIER_TO_MODEL,
        default_model=DEFAULT_MODEL,
        model_provider=TestModelProvider(TestModel.returning_responses([ResponseBuilders.output_message("hi!")])),
    )
    input = _base_input(AUTO_MODEL, [{"role": "user", "content": "hello"}])

    env = ActivityEnvironment()
    with patch(
        "temporal_agent_harness.ai_sdks.openai_agents.jev_auto_routing.classify",
        return_value=_classification("simple"),
    ):
        response = await env.run(activity.invoke_model_activity, input)

    assert response.output[0].content[0].text == "hi!"
