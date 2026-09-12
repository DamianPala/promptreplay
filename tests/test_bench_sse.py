"""Tests for provibench.bench.sse."""

from __future__ import annotations

import json
from typing import Any

from provibench.bench.sse import parse_event_stream


def _sse_bytes(events: list[dict[str, Any]]) -> bytes:
    return ("\n".join(f"data: {json.dumps(e)}" for e in events) + "\n").encode()


def test_parse_event_stream_merges_message_start_and_delta_usage() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "claude-x",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                },
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 42},
        },
    ]
    parsed = parse_event_stream(_sse_bytes(events))
    assert parsed.message_id == "msg_1"
    assert parsed.model == "claude-x"
    assert parsed.stop_reason == "end_turn"
    assert parsed.usage.input_tokens == 100
    assert parsed.usage.cache_read_input_tokens == 20
    assert parsed.usage.cache_creation_input_tokens == 5
    assert parsed.usage.output_tokens == 42
    assert parsed.error is None


def test_parse_event_stream_error_event() -> None:
    events = [{"type": "error", "error": {"type": "overloaded_error", "message": "boom"}}]
    parsed = parse_event_stream(_sse_bytes(events))
    assert parsed.error is not None
    assert "boom" in parsed.error


def test_parse_event_stream_ignores_non_data_and_done_lines() -> None:
    raw = b": comment\n\ndata: [DONE]\n"
    parsed = parse_event_stream(raw)
    assert parsed.message_id is None
    assert parsed.error is None
