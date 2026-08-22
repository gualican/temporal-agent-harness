from __future__ import annotations

import pytest

from temporal_agent_harness.harness.agent_protocol import AcceptedFunction
from temporal_agent_harness.cli.main import _infer_text_field, _pick_handler


def _fn(name: str, parameters: dict) -> AcceptedFunction:
    return AcceptedFunction(name=name, description="", parameters=parameters, output={})


# ---------------------------------------------------------------------------
# _infer_text_field
# ---------------------------------------------------------------------------


def test_infer_text_field_prefers_a_field_literally_named_text() -> None:
    schema = {
        "properties": {"text": {"type": "string"}, "note": {"type": "string"}},
        "required": ["text", "note"],
    }
    assert _infer_text_field(schema) == "text"


def test_infer_text_field_falls_back_to_sole_required_string() -> None:
    schema = {
        "properties": {
            "prompt": {"type": "string"},
            "memory_scope": {"type": "string"},
            "eval_mode": {"type": "string", "default": "none"},
        },
        "required": ["prompt"],
    }
    assert _infer_text_field(schema) == "prompt"


def test_infer_text_field_falls_back_to_sole_string_field_when_none_required() -> None:
    schema = {"properties": {"note": {"type": "string"}}, "required": []}
    assert _infer_text_field(schema) == "note"


def test_infer_text_field_none_when_ambiguous() -> None:
    schema = {
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a", "b"],
    }
    assert _infer_text_field(schema) is None


def test_infer_text_field_none_when_no_string_fields() -> None:
    schema = {"properties": {"count": {"type": "integer"}}, "required": ["count"]}
    assert _infer_text_field(schema) is None


# ---------------------------------------------------------------------------
# _pick_handler
# ---------------------------------------------------------------------------


def test_pick_handler_prefers_ask_by_default() -> None:
    interface = [_fn("slash", {}), _fn("ask", {})]
    assert _pick_handler(interface, None).name == "ask"


def test_pick_handler_falls_back_to_the_only_handler() -> None:
    interface = [_fn("act", {})]
    assert _pick_handler(interface, None).name == "act"


def test_pick_handler_requires_message_type_when_ambiguous() -> None:
    interface = [_fn("act", {}), _fn("slash", {})]
    with pytest.raises(SystemExit, match="pick one with --message-type"):
        _pick_handler(interface, None)


def test_pick_handler_honors_explicit_message_type() -> None:
    interface = [_fn("act", {}), _fn("slash", {})]
    assert _pick_handler(interface, "slash").name == "slash"


def test_pick_handler_rejects_unknown_message_type() -> None:
    interface = [_fn("ask", {})]
    with pytest.raises(SystemExit, match="No 'nope' handler"):
        _pick_handler(interface, "nope")
