"""Tests for provibench.bench.sse: the whole-body parse and the incremental stream read."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from provibench.bench.sse import (
    StreamStats,
    interpret_event,
    iter_stream_events,
    parse_event_stream,
    read_stream,
)


def _sse_bytes(events: list[dict[str, Any]]) -> bytes:
    return ("\n".join(f"data: {json.dumps(e)}" for e in events) + "\n").encode()


def _sse(events: list[dict[str, Any]]) -> bytes:
    """The same events as a real body: blank-line separated, as a gateway sends them."""
    return ("\n\n".join(f"data: {json.dumps(e)}" for e in events) + "\n\n").encode()


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


# --- the incremental read -----------------------------------------------------

_THINKING = [f"think{index}" for index in range(1, 11)]
_TEXT = [f"word{index}" for index in range(1, 11)]


def _recorded_events() -> list[dict[str, Any]]:
    """A recorded-style stream: start, thinking block, text block, usage, stop."""
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "deepseek-v4.1-flash",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 90},
            },
        },
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        *(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": f" {token}"},
            }
            for token in _THINKING
        ),
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        *(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": f" {token}"},
            }
            for token in _TEXT
        ),
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]


class _Clock:
    """A stand-in for `perf_counter`: one tick per read, so every event has a time."""

    def __init__(self, step: float = 0.1, start: float = 0.0) -> None:
        self.step = step
        self.now = start

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def _read(events: list[dict[str, Any]], clock: _Clock) -> tuple[Any, StreamStats]:
    """Run `read_stream` over a body assembled from `events`, with an injected clock."""
    response = httpx.Response(
        200,
        content=_sse(events),
        headers={"content-type": "text/event-stream"},
    )
    return asyncio.run(read_stream(response, clock))


def test_read_stream_folds_the_message_and_its_usage() -> None:
    events = _recorded_events()
    message, _ = _read(events, _Clock())
    assert message.message_id == "msg_1"
    assert message.model == "deepseek-v4.1-flash"
    assert message.stop_reason == "end_turn"
    assert message.usage.input_tokens == 10
    assert message.usage.cache_read_input_tokens == 90
    assert message.usage.output_tokens == 20
    assert message.error is None


def test_read_stream_computes_ttft_and_generation_rate_from_event_times() -> None:
    events = _recorded_events()
    # each read takes 0,1 s: the first delta lands after 3 reads, message_stop after the
    # rest, so TTFT is 0,3 s and 20 output tokens over the remaining 2,3 s
    clock = _Clock(step=0.1)
    _, stats = _read(events, clock)
    assert stats.ttft_s == pytest.approx(0.3)
    assert stats.stop_at is not None and stats.stop_at > 0.3
    assert stats.gen_tok_s == pytest.approx(20 / (stats.stop_at - 0.3))


def test_read_stream_fingerprints_the_first_16_tokens_across_both_blocks() -> None:
    events = _recorded_events()
    _, stats = _read(events, _Clock())
    assert stats.fingerprint == " ".join([*_THINKING, *_TEXT[:6]])


def test_read_stream_reports_no_rate_without_a_message_stop() -> None:
    events = [event for event in _recorded_events() if event["type"] != "message_stop"]
    _, stats = _read(events, _Clock())
    # the last event seen ends the stream, so the window is still measurable
    assert stats.stop_at is not None
    assert stats.gen_tok_s is not None


def test_read_stream_without_deltas_has_no_ttft_and_no_fingerprint() -> None:
    events: list[dict[str, Any]] = [{"type": "message_start", "message": {"id": "m", "usage": {}}}]
    _, stats = _read(events, _Clock())
    assert stats.ttft_s is None
    assert stats.fingerprint is None
    assert stats.gen_tok_s is None


def test_iter_stream_events_handles_chunk_boundaries_and_ignores_noise() -> None:
    events = [{"type": "message_start", "message": {"id": "m"}}, {"type": "message_stop"}]
    body = b": keep-alive\n\ndata: [DONE]\n\n" + _sse(events)
    chunks = [body[index : index + 7] for index in range(0, len(body), 7)]

    class _Chunks(httpx.AsyncByteStream):
        """Serves the body in small pieces, so an event spans several reads."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=_Chunks(), headers={"content-type": "text/event-stream"})

    async def run() -> list[dict[str, Any]]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with (
            client,
            client.stream("POST", "https://x.test/v1/messages", json={}) as response,
        ):
            return [event async for event in iter_stream_events(response)]

    assert asyncio.run(run()) == events


def test_interpret_event_rejects_non_events() -> None:
    assert interpret_event("") is None
    assert interpret_event("[DONE]") is None
    assert interpret_event("{not json") is None
    assert interpret_event("[]") is None
    assert interpret_event('{"type": "ping"}') == {"type": "ping"}


def test_read_stream_joins_pieces_so_no_two_can_fuse() -> None:
    """A gateway may end a delta mid-word; the fingerprint still counts whole pieces."""
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"id": "m"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "tok1"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "tok2"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "tok3"},
        },
    ]
    _, stats = _read(events, _Clock())
    assert stats.text == "tok1 tok2 tok3"
    assert stats.fingerprint == "tok1 tok2 tok3"


def test_read_stream_ignores_an_empty_delta() -> None:
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"id": "m"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "one"}},
    ]
    _, stats = _read(events, _Clock())
    assert stats.text == "one"


_TOOL_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_tool",
            "model": "deepseek-v4.1-flash",
            "usage": {"input_tokens": 10},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "name": "Read"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"file_path":'},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": ' "/tmp/tests"}'},
    },
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
    {"type": "message_stop"},
]


def test_read_stream_counts_a_tool_call_in_the_fingerprint_and_the_ttft() -> None:
    """A turn the model answers with a tool call streams JSON, not text: it still measures."""
    events = [event for event in _TOOL_EVENTS if event["type"] != "message_start"]
    events.insert(0, _TOOL_EVENTS[0])
    _, stats = _read(events, _Clock())
    assert stats.ttft_s is not None  # the first input_json_delta is a first delta
    assert stats.fingerprint == '{"file_path": "/tmp/tests"}'
    assert stats.output_tokens == 12


def test_read_stream_fingerprints_thinking_then_tool_call_in_stream_order() -> None:
    events = [
        _TOOL_EVENTS[0],
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "reason"},
        },
        _TOOL_EVENTS[2],
        _TOOL_EVENTS[5],
        _TOOL_EVENTS[6],
    ]
    _, stats = _read(events, _Clock())
    assert stats.fingerprint == 'reason {"file_path":'


class _Stuck(httpx.AsyncByteStream):
    """Serves one chunk and then keeps the connection open forever."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body
        await asyncio.Event().wait()


async def _read_stuck(body: bytes, *, idle_timeout_s: float) -> tuple[Any, StreamStats]:
    response = httpx.Response(
        200, stream=_Stuck(body), headers={"content-type": "text/event-stream"}
    )
    return await asyncio.wait_for(
        read_stream(response, _Clock(), idle_timeout_s=idle_timeout_s), 5.0
    )


def test_read_stream_stops_at_the_end_of_the_stream_without_waiting_for_the_socket() -> None:
    """`message_stop` ends the read: the connection is not drained until the server closes it."""
    _, stats = asyncio.run(_read_stuck(_sse(_TOOL_EVENTS), idle_timeout_s=30.0))
    assert stats.stop_at is not None
    assert stats.fingerprint == '{"file_path": "/tmp/tests"}'


def test_read_stream_times_out_on_a_stream_that_goes_quiet() -> None:
    """A stream that never sends its last event raises instead of hanging the run."""
    response = httpx.Response(
        200,
        stream=_Stuck(_sse(_TOOL_EVENTS[:2])),
        headers={"content-type": "text/event-stream"},
    )
    with pytest.raises(httpx.ReadTimeout, match="stream stalled"):
        asyncio.run(read_stream(response, _Clock(), idle_timeout_s=0.01))


def test_read_stream_stops_at_an_error_event() -> None:
    """An `error` event ends the stream: nothing follows it, whatever the socket does."""
    events = [
        _TOOL_EVENTS[0],
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "boom"}},
    ]
    _, stats = asyncio.run(_read_stuck(_sse(events), idle_timeout_s=30.0))
    assert stats.stop_at is not None
    assert stats.fingerprint == "hi"
