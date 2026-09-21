"""Tests for the probe protocol: nonce, rungs, skip, retries, and the aggregation.

Everything here runs against `httpx.MockTransport`; the only boundary replaced is the
network `httpx.AsyncClient` opens, never the probe logic itself.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from provibench.bench.estimate import SpecPrices
from provibench.bench.labels import column_labels
from provibench.bench.nonce import inject_nonce, probe_nonce, require_stampable, run_nonce
from provibench.bench.probe import (
    ProbeOptions,
    ProbeResult,
    ProbeRun,
    is_failed,
    is_served,
    run_probe,
)
from provibench.bench.probe_drift import apply_drift
from provibench.bench.probe_models import Role
from provibench.bench.probe_stream import BURST_NOTE, StreamResult
from provibench.bench.probe_summary import (
    ProbeSummary,
    TtlRead,
    cached_of,
    summarize_probe,
    summarize_rung,
)
from provibench.bench.probe_tables import probe_blocks, probe_labels, probe_markdown, render_probe
from provibench.bench.rungs import (
    broadcast_repeats,
    parse_int_list,
    select_rungs,
    validate_rungs,
)
from provibench.bench.sse import ParsedMessage, StreamStats
from provibench.bench.summary import cache_mode_note, cache_mode_sentence
from provibench.bench.targets import Prices, RunSpec, Target, parse_run_spec
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage
from provibench.commands.probe_spend import session_footer_lines

_RealAsyncClient = httpx.AsyncClient
_RUN_HEX = "0123456789ab"
_HEADERS = {"anthropic-version": "2023-06-01"}


def _target(kind: Literal["openrouter", "anthropic"] = "anthropic") -> Target:
    return Target(
        name="fake", url="https://api.test/v1/messages", api_key_env="FAKE_KEY", kind=kind
    )


def _spec(model: str = "model-a", *, providers: list[str] | None = None) -> RunSpec:
    kind: Literal["openrouter", "anthropic"] = "openrouter" if providers else "anthropic"
    return RunSpec(target=_target(kind), model=model, providers=providers or [])


def _system() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "You are helpful."},
        {"type": "text", "text": "Repo context.", "cache_control": {"type": "ephemeral"}},
    ]


def _entry(
    seq: int, pairs: int = 2, *, prompt_tokens: int = 100, thinking: bool = False
) -> TraceEntry:
    """A recorded turn: a system prompt, a growing conversation, and recorded usage."""
    messages: list[dict[str, Any]] = []
    for index in range(pairs):
        messages.append({"role": "user", "content": f"question {index}"})
        messages.append({"role": "assistant", "content": f"answer {index}"})
    body: dict[str, Any] = {
        "model": "orig-model",
        "max_tokens": 1024,
        "stream": True,
        "system": _system(),
        "messages": messages,
    }
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    return TraceEntry(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        path="/v1/messages",
        headers=dict(_HEADERS),
        body=body,
        conversation="c1",
        response=RecordedResponse(
            status=200, latency_ms=1.0, usage=Usage(input_tokens=prompt_tokens)
        ),
    )


def _trace(count: int = 4) -> list[TraceEntry]:
    return [_entry(seq, pairs=seq) for seq in range(1, count + 1)]


def _ok(cached: int = 0, input_tokens: int = 100) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cached,
                "output_tokens": 1,
            },
        },
    )


_STREAM_TOKENS = [f"tok{index}" for index in range(1, 19)]
"""The throughput fixture's output: 18 whitespace-separated pieces, so 16 is a real cut."""


def _sse_bytes(events: list[dict[str, Any]]) -> bytes:
    return ("\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\n").encode()


def _stream_response(
    *, cached: int = 0, input_tokens: int = 100, model: str = "model-a"
) -> httpx.Response:
    """A streamed generation: message_start, a thinking block, deltas, usage, message_stop."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_read_input_tokens": cached,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    ]
    events.extend(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": f" {token}"},
        }
        for token in _STREAM_TOKENS[:8]
    )
    events.append({"type": "content_block_stop", "index": 0})
    events.append(
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}
    )
    events.extend(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": f" {token}"},
        }
        for token in _STREAM_TOKENS[8:]
    )
    events.append({"type": "content_block_stop", "index": 1})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 18},
        }
    )
    events.append({"type": "message_stop"})
    return httpx.Response(
        200,
        content=_sse_bytes(events),
        headers={"content-type": "text/event-stream"},
    )


def _client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.AsyncClient]:
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)  # type: ignore[arg-type]

    return factory


def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    specs: list[RunSpec],
    entries: list[TraceEntry],
    options: ProbeOptions,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    on_progress: Callable[[ProbeResult], None] | None = None,
) -> ProbeRun:
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    return asyncio.run(
        run_probe(specs, entries, options, {"FAKE_KEY": "secret"}, on_progress=on_progress)
    )


def _record(  # noqa: PLR0913 (a factory mirroring the record model's fields)
    role: Role,
    attempt: int,
    *,
    rung: int = 1,
    status: int = 200,
    cached: int = 0,
    prompt: int = 100,
    cache_write: int = 0,
    native_cached: int | None = None,
    generation: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    error: str | None = None,
    latency: float = 10.0,
) -> ProbeResult:
    return ProbeResult(
        spec_label="fake:model-a",
        rung=rung,
        role=role,
        attempt=attempt,
        seq=1,
        status=status,
        latency_ms=latency,
        provider=provider,
        model=model,
        prompt_total=prompt,
        cached=cached,
        cache_write=cache_write,
        native_tokens_cached=native_cached,
        generation=generation,
        error=error,
    )


def _ref(kind: str) -> Any:
    """A stand-in spec reference for `apply_drift`: only its kind and providers are read."""

    class _Ref:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.providers: list[str] = []

    return _Ref(kind)


def _prices() -> SpecPrices:
    return SpecPrices(
        prices=Prices(input=1.0, cache_read=0.1, cache_write=1.0, output=2.0), source="targets"
    )


# --- nonce --------------------------------------------------------------------


def test_inject_nonce_stamps_the_first_system_block_and_keeps_cache_control() -> None:
    entry = _entry(1)
    stamped = inject_nonce(entry.body, "N")
    assert stamped["system"][0]["text"] == "N\nYou are helpful."
    assert stamped["system"][1] == {
        "type": "text",
        "text": "Repo context.",
        "cache_control": {"type": "ephemeral"},
    }
    assert entry.body["system"][0]["text"] == "You are helpful."  # the recording is untouched


def test_inject_nonce_prefixes_a_string_system_and_copies_the_body() -> None:
    body: dict[str, Any] = {"system": "hello", "messages": []}
    stamped = inject_nonce(body, "N")
    assert stamped["system"] == "N\nhello"
    assert body["system"] == "hello"


def test_inject_nonce_rejects_a_body_it_cannot_stamp() -> None:
    with pytest.raises(ValueError, match="no 'system'"):
        inject_nonce({"messages": []}, "N")
    with pytest.raises(ValueError, match="no 'system'"):
        inject_nonce({"system": None}, "N")
    with pytest.raises(ValueError, match="non-empty list"):
        inject_nonce({"system": []}, "N")
    with pytest.raises(ValueError, match="no 'text'"):
        inject_nonce({"system": [{"type": "text"}]}, "N")


def test_require_stampable_names_the_turn_and_the_way_out() -> None:
    require_stampable([("turn 1", _entry(1).body)])
    with pytest.raises(ValueError, match="turn 2 has no system prompt"):
        require_stampable([("turn 1", _entry(1).body), ("turn 2", {"messages": []})])
    with pytest.raises(ValueError, match="--warm"):
        require_stampable([("turn 2", {"messages": []})])


def test_nonces_derive_from_the_run_hex() -> None:
    assert run_nonce("abc123") == "provibench-run:abc123"
    assert probe_nonce("abc123", 13, "fake:a") == "provibench-probe:abc123:13:fake:a"


def test_probe_nonce_differs_by_endpoint_not_just_by_rung() -> None:
    # two OpenRouter tags of one provider probed in the same run must never send the same
    # bytes, or the second one's "cold" write reads back the first one's cache
    a = probe_nonce("abc123", 1, "or:model@sail-research/us")
    b = probe_nonce("abc123", 1, "or:model@sail-research/fp8")
    assert a != b
    assert a == probe_nonce("abc123", 1, "or:model@sail-research/us")  # same endpoint, same
    assert a.startswith("provibench-probe:abc123:1:")


# --- rung selection -----------------------------------------------------------


def test_parse_int_list_accepts_spaces_and_rejects_empty() -> None:
    assert parse_int_list("1, 13 ,30") == [1, 13, 30]
    with pytest.raises(ValueError, match="comma-separated list"):
        parse_int_list(" , ")
    with pytest.raises(ValueError, match="comma-separated list"):
        parse_int_list("1,x")


def test_select_rungs_picks_smallest_middle_and_largest_by_recorded_prompt() -> None:
    conversation = [
        _entry(1, prompt_tokens=50),
        _entry(2, prompt_tokens=10),
        _entry(3, prompt_tokens=40),
        _entry(4, prompt_tokens=30),
        _entry(5, prompt_tokens=20),
    ]
    # candidates are turns 1-4; by size: 2 (10), 4 (30), 3 (40), 1 (50)
    assert select_rungs(conversation) == [1, 2, 3]


def test_select_rungs_dedupes_a_two_turn_conversation() -> None:
    assert select_rungs(_trace(count=2)) == [1]


def test_select_rungs_falls_back_to_the_body_size_when_unrecorded() -> None:
    conversation = [_entry(seq, pairs=seq) for seq in range(1, 5)]
    for entry in conversation:
        entry.response = None
    assert select_rungs(conversation) == [1, 2, 3]


def test_select_rungs_rejects_a_trace_without_a_next_turn() -> None:
    with pytest.raises(ValueError, match="no turn with a following turn"):
        select_rungs(_trace(count=1))


def test_broadcast_repeats_repeats_the_last_value_for_missing_entries() -> None:
    assert broadcast_repeats([1, 13, 30], [5, 2]) == [5, 2, 2]
    assert broadcast_repeats([1, 2], [3, 4, 5]) == [3, 4]
    with pytest.raises(ValueError, match="at least one value"):
        broadcast_repeats([1], [])


def test_validate_rungs_rejects_a_rung_without_a_next_turn() -> None:
    validate_rungs([1, 3], _trace())
    with pytest.raises(ValueError, match="rung 4 needs turn 5"):
        validate_rungs([4], _trace())
    with pytest.raises(ValueError, match="at least 1"):
        validate_rungs([0], _trace())
    with pytest.raises(ValueError, match="no conversations"):
        validate_rungs([1], [])


def test_probe_options_resolved_broadcasts_and_keeps_explicit_rungs() -> None:
    options = ProbeOptions(rungs=[1, 3], repeats=[5, 2])
    resolved = options.resolved(_trace())
    assert resolved.rungs == [1, 3]
    assert resolved.repeats == [5, 2]

    default = ProbeOptions().resolved(_trace())
    assert default.rungs == [1, 2, 3]
    assert default.repeats == [6, 2, 2]


def test_probe_options_resolved_collapses_a_repeated_rung() -> None:
    # a rung probed twice would have its second cold request read the first one's cache
    resolved = ProbeOptions(rungs=[3, 1, 3], repeats=[2]).resolved(_trace())
    assert resolved.rungs == [1, 3]
    assert resolved.repeats == [2, 2]


def test_probe_options_resolved_rejects_an_impossible_rung() -> None:
    with pytest.raises(ValueError, match="rung 4 needs turn 5"):
        ProbeOptions(rungs=[4]).resolved(_trace())


# --- the run ------------------------------------------------------------------


def test_run_probe_sends_cold_then_warm_with_one_nonce_per_rung_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok()

    targets = {"fake": _target()}
    specs = [parse_run_spec("fake:a", targets), parse_run_spec("fake:b", targets)]
    run = _run_probe(
        monkeypatch,
        specs,
        _trace(),
        ProbeOptions(rungs=[1, 3], repeats=[1, 1], gap_s=0.0, throughput=False),
        handler,
    )

    assert run.options.repeats == [1, 1]
    assert [(r.spec_label, r.rung, r.role, r.attempt) for r in run.records["fake:a"]] == [
        ("fake:a", 1, "cold", 0),
        ("fake:a", 1, "warm", 1),
        ("fake:a", 3, "cold", 0),
        ("fake:a", 3, "warm", 1),
    ]
    nonces: list[str] = []
    by_rung_and_label: dict[tuple[int, str], set[str]] = {}
    for body in bodies:
        nonce = body["system"][0]["text"].splitlines()[0]
        nonces.append(nonce)
        rung, label = int(nonce.split(":")[2]), nonce.split(":", 3)[3]
        by_rung_and_label.setdefault((rung, label), set()).add(nonce)
    assert set(by_rung_and_label) == {
        (1, "fake:a"),
        (3, "fake:a"),
        (1, "fake:b"),
        (3, "fake:b"),
    }
    # one nonce per rung and endpoint: every request of one rung of one endpoint agrees
    assert all(len(group) == 1 for group in by_rung_and_label.values())
    [rung1_a], [rung3_a], [rung1_b] = (
        by_rung_and_label[1, "fake:a"],
        by_rung_and_label[3, "fake:a"],
        by_rung_and_label[1, "fake:b"],
    )
    assert rung1_a != rung3_a  # rung 3 cannot read rung 1's write
    assert rung1_a != rung1_b  # "fake:b" cannot read what "fake:a" wrote at the same rung
    assert all(nonce.startswith(f"provibench-probe:{run.run_hex}:") for nonce in nonces)


def test_run_probe_gives_two_endpoints_of_one_run_and_rung_different_nonces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two OpenRouter tags of one provider, probed one after another in the same run, must
    never send the same bytes: the second tag's "cold" write would otherwise be a full cache
    hit of the first tag's write, and the report would print it as unrealistically cheap."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok()

    us = _spec("model-a", providers=["sail-research/us"])
    fp8 = _spec("model-a", providers=["sail-research/fp8"])
    _run_probe(
        monkeypatch,
        [us, fp8],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0, throughput=False),
        handler,
    )
    first_nonce = bodies[0]["system"][0]["text"].splitlines()[0]
    second_nonce = bodies[2]["system"][0]["text"].splitlines()[0]  # the other spec's cold write
    assert first_nonce != second_nonce


def test_run_probe_skips_a_rung_whose_cold_request_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        httpx.Response(500, json={"error": {"message": "boom"}}),  # rung 1 cold fails
        _ok(),
        _ok(),
        _ok(),  # rung 3: cold + two warm
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return response

    run = _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1, 3], repeats=[3, 2], gap_s=0.0, throughput=False),
        handler,
    )

    assert [(r.rung, r.role) for r in run.records["fake:model-a"]] == [
        (1, "cold"),
        (3, "cold"),
        (3, "warm"),
        (3, "warm"),
    ]
    assert calls["n"] == 4  # rung 1 sent no warm reads


def test_warm_attempts_send_byte_identical_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return _ok()

    _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[3], gap_s=0.0, throughput=False),
        handler,
    )

    cold, *warm = sent
    assert len(warm) == 3
    assert len(set(warm)) == 1  # the repeats are the same bytes, not just equal objects
    assert cold != warm[0]  # turn k and turn k+1 are different requests
    parsed = [json.dumps(json.loads(body), sort_keys=True) for body in sent]
    assert len(set(parsed[1:])) == 1


def test_warm_mode_sends_no_nonce(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok()

    _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0, warm=True, throughput=False),
        handler,
    )
    assert [body["system"][0]["text"] for body in bodies] == ["You are helpful."] * 2


def test_run_probe_reports_one_progress_line_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, str, int]] = []

    def on_progress(record: ProbeResult) -> None:
        seen.append((record.rung, record.role, record.attempt))

    _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[2], gap_s=0.0, throughput=False),
        lambda request: _ok(),
        on_progress=on_progress,
    )
    assert seen == [(1, "cold", 0), (1, "warm", 1), (1, "warm", 2)]


def test_gap_is_waited_before_every_warm_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[3], gap_s=0.5),
        lambda r: _ok(),
    )
    assert sleeps == [0.5, 0.5, 0.5]


def test_send_retries_429_and_503_with_the_backoff_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    responses = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(503, json={"error": {"message": "overloaded"}}),
        _ok(cached=50),
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return response

    run = _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0, throughput=False),
        handler,
    )
    [cold] = run.records["fake:model-a"]
    assert cold.status == 200
    assert cold.retries == 2
    assert cold.cached == 50
    assert sleeps == [2.0, 5.0]


def test_send_does_not_retry_a_500(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    run = _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0, throughput=False),
        handler,
    )
    [cold] = run.records["fake:model-a"]
    assert calls["n"] == 1
    assert cold.retries == 0
    assert cold.status == 500
    assert not is_served(cold)
    assert is_failed(cold)


_THINKING_400 = "max_tokens must be greater than thinking.budget_tokens"


def _thinking_transport(sent: list[bytes]) -> Callable[[httpx.Request], httpx.Response]:
    """A provider that 400s any request carrying a `thinking` parameter, as replay clients see."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        body = json.loads(request.content)
        if "thinking" in body:
            return httpx.Response(400, json={"error": {"message": _THINKING_400}})
        return _ok(cached=50)

    return handler


def test_send_retries_a_400_about_thinking_without_the_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[bytes] = []
    run = _run_probe(
        monkeypatch,
        [_spec()],
        [_entry(1, thinking=True), _entry(2, thinking=True)],
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0, throughput=False),
        _thinking_transport(sent),
    )

    cold, warm = run.records["fake:model-a"]
    assert cold.status == 200
    assert cold.retries == 0  # the thinking retry is not a 429/503 retry
    assert cold.note == "thinking param dropped"
    assert warm.note == "thinking param dropped"
    # the cold request dropped `thinking`, so the warm body never carried it
    assert ["thinking" in json.loads(body) for body in sent] == [True, False, False]


def test_warm_bodies_stay_byte_identical_after_a_thinking_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[bytes] = []
    _run_probe(
        monkeypatch,
        [_spec()],
        [_entry(1, thinking=True), _entry(2, thinking=True), _entry(3)],
        ProbeOptions(rungs=[1], repeats=[3], gap_s=0.0, throughput=False),
        _thinking_transport(sent),
    )

    cold, *warm = sent
    assert "thinking" in json.loads(cold)
    warm_bodies = [json.dumps(json.loads(body), sort_keys=True) for body in warm]
    assert len(set(warm_bodies)) == 1  # every repeat sent the same bytes as every other
    assert all("thinking" not in json.loads(body) for body in warm)


def test_a_thinking_drop_on_a_warm_read_applies_to_the_later_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # only turn 2's recorded body carries `thinking`, so the cold request succeeds with it
    sent: list[bytes] = []
    run = _run_probe(
        monkeypatch,
        [_spec()],
        [_entry(1), _entry(2, thinking=True), _entry(3), _entry(4)],
        ProbeOptions(rungs=[1], repeats=[3], gap_s=0.0, throughput=False),
        _thinking_transport(sent),
    )

    all_warm = run.records["fake:model-a"][1:]
    assert [record.status for record in all_warm] == [200, 200, 200]
    assert [record.note for record in all_warm] == [
        "thinking param dropped",
        "thinking param dropped",
        "thinking param dropped",
    ]
    # the cold, the first warm and its retry, then two repeats that inherit the drop
    assert len(sent) == 5
    assert ["thinking" in json.loads(body) for body in sent[1:]] == [True, False, False, False]
    repeated = [json.dumps(json.loads(body), sort_keys=True) for body in sent[2:]]
    assert len(set(repeated)) == 1


def test_send_does_not_retry_a_400_without_the_thinking_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "invalid api key"}})

    run = _run_probe(
        monkeypatch,
        [_spec()],
        [_entry(1, thinking=True), _entry(2)],
        ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0, throughput=False),
        handler,
    )
    [cold] = run.records["fake:model-a"]
    assert calls["n"] == 1
    assert cold.status == 400
    assert cold.note is None
    assert not is_served(cold)


def test_run_probe_enriches_openrouter_records_from_the_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints = {
        "data": {
            "endpoints": [
                {
                    "provider_name": "Novita",
                    "tag": "novita",
                    "pricing": {
                        "prompt": "0.0000001",
                        "completion": "0.0000002",
                        "input_cache_read": "0.00000001",
                        "input_cache_write": "0.0000001",
                    },
                }
            ]
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=endpoints)
        if request.url.path.endswith("/generation"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "gen_1",
                        "provider_name": "Novita",
                        "native_tokens_prompt": 100,
                        "native_tokens_cached": 45,
                        "total_cost": 0.0007,
                    }
                },
            )
        return _ok(cached=0, input_tokens=100)

    spec = _spec("deepseek/model", providers=["novita"])
    run = _run_probe(
        monkeypatch,
        [spec],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0, throughput=False),
        handler,
    )
    cold, warm = run.records[spec.label]
    assert [record.provider for record in (cold, warm)] == ["Novita", "Novita"]
    assert [record.native_tokens_cached for record in (cold, warm)] == [45, 45]
    assert cold.cached == 0  # the response reported no cache; the fallback is the gateway's

    assert cold.generation is not None
    assert cold.generation["id"] == "gen_1"  # the raw lookup is kept for costing

    summary = summarize_probe(spec.label, [cold, warm])
    assert summary.hit_rate == 1.0
    assert summary.billed_usd == pytest.approx(0.0014)
    assert (
        "did not report the cached share for 2 requests; using OpenRouter's own billing "
        "record instead." in summary.notes
    )


# --- aggregation --------------------------------------------------------------


def test_summarize_rung_hit_rate_fractions_and_cache_write() -> None:
    records = [
        _record("cold", 0, cached=0, prompt=100, cache_write=50, latency=20.0),
        _record("warm", 1, cached=0, latency=30.0),
        _record("warm", 2, cached=98, latency=40.0),
        _record("warm", 3, cached=200, latency=50.0),
    ]
    rung = summarize_rung("fake:model-a", 1, records)
    assert rung.prompt_cold == 100
    assert rung.cached_cold == 0
    assert rung.hit_rate == pytest.approx(2 / 3)
    assert rung.cached_fraction == pytest.approx((0.98 + 1.0) / 2)  # capped at 1.0 per hit
    assert rung.hits == [0.0, 0.98, 1.0]
    assert rung.cache_write_cold == 50
    assert rung.cold_ms == pytest.approx(20.0)
    assert rung.warm_ms == pytest.approx(40.0)  # median over the served attempts
    assert rung.skipped is False
    assert rung.cold_error is None
    assert rung.errors == 0


def test_summarize_rung_marks_failed_warm_attempts_and_ignores_them_in_the_median() -> None:
    records = [
        _record("cold", 0),
        _record("warm", 1, status=500, error="boom", latency=5.0),
        _record("warm", 2, cached=10, latency=100.0),
        _record("warm", 3, cached=10, latency=300.0),
    ]
    rung = summarize_rung("fake:model-a", 1, records)
    assert rung.hits == [None, 0.1, 0.1]
    assert rung.errors == 1
    assert rung.hit_rate == 1.0  # only the served attempts are the denominator
    assert rung.warm_ms == pytest.approx(200.0)


def test_summarize_rung_skips_a_rung_whose_cold_request_failed() -> None:
    records = [_record("cold", 0, status=503, error="down")]
    rung = summarize_rung("fake:model-a", 1, records)
    assert rung.skipped is True
    assert rung.cold_error == "down"
    assert rung.cached_fraction is None
    assert rung.hits == []
    assert rung.hit_rate == 0.0
    assert rung.warm_ms is None
    assert rung.errors == 1


_GUARDRAIL_ERROR = (
    '{"type": "not_found_error", "message": "0 endpoints out of 1 requested are available'
    " matching your guardrail restrictions and data policy. We removed them for the following"
    " reasons (...):\\nPaid model training violation (account settings): 1 endpoint excluded;"
    ' configurable at https://openrouter.ai/settings/privacy", "error_type": "not_found"}'
)
"""The live refusal: the endpoint is not served to this account at all, in under 300 ms."""


def _cold_failure(error: str, rung: int) -> ProbeResult:
    return _record("cold", 0, rung=rung, status=404, error=error)


def test_a_spec_whose_every_cold_failed_says_why() -> None:
    """Three 404s and a reason: the row shows `errors 3`, and the note shows the setting."""
    records = [_cold_failure(_GUARDRAIL_ERROR, rung) for rung in (1, 13, 23)]
    summary = summarize_probe("or:model@alibaba", records)
    assert summary.skipped == 3
    assert "skipped, not_found: Paid model training violation (account settings)" in summary.notes
    assert (
        "or:model@alibaba: skipped, not_found: Paid model training violation (account settings)"
        in render_probe([summary])
    )


def test_a_rate_limited_spec_is_skipped_with_its_error_type() -> None:
    error = (
        '{"type": "rate_limit_exceeded", "message": "slow down",'
        ' "error_type": "rate_limit_exceeded"}'
    )
    records = [_cold_failure(error, rung) for rung in (1, 13, 23)]
    summary = summarize_probe("or:model@alibaba", records)
    assert "skipped, rate_limit_exceeded" in summary.notes


def test_a_spec_that_answered_one_rung_is_not_called_skipped() -> None:
    """One cold that went through means the endpoint serves this account; the rest is noise."""
    records = [
        _record("cold", 0, rung=1),
        _cold_failure(_GUARDRAIL_ERROR, 13),
        _cold_failure(_GUARDRAIL_ERROR, 23),
    ]
    summary = summarize_probe("or:model@alibaba", records)
    assert not [note for note in summary.notes if note.startswith("skipped")]


def test_a_spec_that_failed_differently_each_time_is_not_called_skipped() -> None:
    records = [
        _cold_failure(_GUARDRAIL_ERROR, 1),
        _cold_failure('{"type": "server_error"}', 2),
    ]
    summary = summarize_probe("or:model@alibaba", records)
    assert not [note for note in summary.notes if note.startswith("skipped")]


def test_a_404_body_of_the_live_shape_becomes_the_records_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path from a real refusal body to the note, through the probe's own parsing."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            404,
            json={
                "type": "not_found_error",
                "message": "0 endpoints out of 1 requested are available",
                "error_type": "not_found",
            },
        )

    run = _run_probe(
        monkeypatch, [_spec()], _trace(), ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0), handler
    )
    [cold] = [record for record in run.records["fake:model-a"] if record.role == "cold"]
    assert "not_found" in (cold.error or "")
    [summary] = [summarize_probe("fake:model-a", run.records["fake:model-a"])]
    assert "skipped, not_found" in " ".join(summary.notes)


def test_a_transport_error_is_not_a_kind_of_refusal() -> None:
    """A timeout is not the provider saying no, so the note cannot name an error type."""
    records = [_cold_failure("ReadTimeout(...)", rung) for rung in (1, 2)]
    summary = summarize_probe("or:model@alibaba", records)
    assert not [note for note in summary.notes if note.startswith("skipped")]


def test_summarize_probe_pools_the_rungs_and_prices_the_result() -> None:
    records = [
        _record("cold", 0, rung=1, prompt=100, cache_write=100),
        _record("warm", 1, rung=1, cached=100),
        _record("warm", 2, rung=1, cached=0),
        _record("cold", 0, rung=2, prompt=200),
        _record("warm", 1, rung=2, cached=100),
    ]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    assert [rung.rung for rung in summary.rungs] == [1, 2]
    assert summary.hit_rate == pytest.approx(2 / 3)  # 2 hits over 3 served warm reads
    assert summary.cached_fraction == pytest.approx((1.0 + 0.5) / 2)
    assert summary.h == pytest.approx(2 / 3 * 0.75)
    # Both rungs' attempt-1 read hit (rung 1 fully, rung 2 half); the first-read figures
    # sit next to the pooled ones, and the price is weighted by the pooled h, the steady
    # state a long session runs in.
    assert summary.first_hit_rate == pytest.approx(1.0)
    assert summary.first_cached_fraction == pytest.approx((1.0 + 0.5) / 2)
    assert summary.first_h == pytest.approx(0.75)
    pooled_h = 2 / 3 * 0.75
    assert summary.eff_per_m_prompt == pytest.approx((1 - pooled_h) * 1.0 + pooled_h * 0.1)
    assert summary.price_source == "targets"
    assert summary.errors == 0
    assert summary.skipped == 0


def test_summarize_probe_prices_from_pooled_reads_when_no_first_read_was_served() -> None:
    """Every rung's attempt-1 read failed; the only served warm reads are repeats. The
    first-read figures are absent and the pooled price is still computed."""
    records = [
        _record("cold", 0, rung=1, prompt=100),
        _record("warm", 1, rung=1, status=500, error="boom"),
        _record("warm", 2, rung=1, cached=100),
    ]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    assert summary.first_hit_rate is None
    assert summary.first_cached_fraction is None
    assert summary.first_h is None
    assert summary.h == pytest.approx(1.0)  # the one served warm read, a full hit
    assert summary.eff_per_m_prompt == pytest.approx((1 - 1.0) * 1.0 + 1.0 * 0.1)
    assert not [note for note in summary.notes if note.startswith("eff $/M")]


def test_summarize_probe_eff_per_m_equals_pooled_with_one_repeat_per_rung() -> None:
    """With one warm read per rung (`--repeats 1`), the first read is the only read, so the
    first-read figures and the pooled ones the price uses are the same by construction."""
    records = [
        _record("cold", 0, rung=1, prompt=100),
        _record("warm", 1, rung=1, cached=40),
        _record("cold", 0, rung=2, prompt=200),
        _record("warm", 1, rung=2, cached=200),
    ]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    assert summary.first_h == pytest.approx(summary.h)
    assert summary.eff_per_m_prompt == pytest.approx(
        (1 - summary.h) * 1.0 + summary.h * 0.1  # type: ignore[operator]
    )


def test_summarize_probe_reports_a_skipped_rung_and_no_hit_rate_without_warm_reads() -> None:
    records = [_record("cold", 0, rung=1, status=500, error="boom")]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    assert summary.skipped == 1
    assert summary.hit_rate is None  # nothing was served, so there is no rate to report
    assert summary.h is None
    assert summary.eff_per_m_prompt is None


def test_summarize_probe_caps_the_cached_fraction_and_falls_back_to_native_cached() -> None:
    records = [
        _record("cold", 0, prompt=100),
        # the response reports nothing, but the gateway says the whole prefix was cached
        _record("warm", 1, cached=0, native_cached=120),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert summary.hit_rate == 1.0
    assert summary.cached_fraction == 1.0  # 120 / 100 is capped
    assert cached_of(records[1]) == 120
    assert (
        "did not report the cached share for 1 request; using OpenRouter's own billing "
        "record instead." in summary.notes
    )


def test_summarize_probe_prefers_the_response_usage_over_the_native_count() -> None:
    record = _record("warm", 1, cached=40, native_cached=999)
    assert cached_of(record) == 40


def test_summarize_probe_sums_the_billed_cost_when_the_generation_has_one() -> None:
    records = [
        _record("cold", 0, generation={"total_cost": 0.001}),
        _record("warm", 1, generation={"total_cost": 0.002}),
        _record("warm", 2),
    ]
    summary = summarize_probe("fake:model-a", records, notes=["endpoint lookup failed"])
    assert summary.billed_usd == pytest.approx(0.003)
    assert summary.notes[0] == "endpoint lookup failed"


def test_summarize_probe_counts_errors_retries_and_providers() -> None:
    records = [
        _record("cold", 0, provider="Novita"),
        _record("warm", 1, status=500, error="boom", provider="Novita"),
    ]
    records[1].retries = 2
    summary = summarize_probe("fake:model-a", records)
    assert summary.errors == 1
    assert summary.retries == 2
    assert summary.providers_seen == {"Novita": 1}


# --- rendering ----------------------------------------------------------------


def test_render_probe_without_summaries_is_two_headers() -> None:
    assert render_probe([]).split("\n\n")[0].startswith("endpoint")
    assert render_probe([]).split("\n\n")[1].startswith("endpoint | rung")


def test_render_probe_has_an_endpoint_table_and_a_rung_table_within_130_columns() -> None:
    # The `cache $/M` column (item 2) widens the endpoint table past the old 120-column
    # budget by a fixed amount no label planning can claw back; 130 is the new ceiling.
    records = [
        _record("cold", 0, prompt=100),
        _record("warm", 1, cached=0),
        _record("warm", 2, status=503, error="down"),
        _record("warm", 3, cached=100),
    ]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    blocks = [block.splitlines() for block in render_probe([summary]).split("\n\n")]
    endpoint_table, rung_table = blocks[0], blocks[1]
    endpoint_header = [cell.strip() for cell in endpoint_table[0].split("|")]
    assert endpoint_header[:3] == ["endpoint", "hit %", "1st hit %"]
    assert endpoint_table[2].startswith("fake:model-a")
    rung_header = [cell.strip() for cell in rung_table[0].split("|")]
    assert rung_header[:5] == ["endpoint", "rung", "prompt", "cached cold", "hits"]
    assert "0 x 1" in rung_table[2]  # one cell per warm attempt, `x` for the failed one
    assert all(len(line) <= 130 for block in blocks for line in block)


def test_run_probe_streams_the_throughput_request_after_the_warm_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One streamed request per rung, after its warm reads, recorded with its fingerprint."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            calls.append("stream")
            return _stream_response(cached=40)
        calls.append("read")
        return _ok(cached=40, input_tokens=60)

    run = _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[2], gap_s=0.0),
        handler,
    )

    assert calls == ["read", "read", "read", "stream"]  # cold, warm, warm, then the stream
    assert [
        (record.rung, record.role, record.attempt) for record in run.records["fake:model-a"]
    ] == [
        (1, "cold", 0),
        (1, "warm", 1),
        (1, "warm", 2),
        (1, "stream", 0),
    ]
    [stream] = [record for record in run.records["fake:model-a"] if record.role == "stream"]
    assert stream.fingerprint == " ".join(_STREAM_TOKENS[:16])  # thinking first, then text
    assert stream.usage.output_tokens == 18
    assert stream.cached == 40
    assert stream.gen_tok_s is None or stream.gen_tok_s > 0


def _stream_result(
    *, latency_ms: float, ttft_ms: float, deltas: int, output_tokens: int
) -> StreamResult:
    """A streamed result with the timing and the arrival shape a test wants."""
    stats = StreamStats()
    stats.first_delta_at = ttft_ms / 1000
    stats.stop_at = latency_ms / 1000
    stats.output_tokens = output_tokens
    stats.deltas = deltas
    return StreamResult(
        message=ParsedMessage(), stats=stats, status=200, latency_ms=latency_ms, started=0.0
    )


def test_a_burst_stream_has_no_generation_rate() -> None:
    """The live shape: 97 tokens 9 ms after the first one is a flush, not a generation."""
    burst = _stream_result(latency_ms=3103.0, ttft_ms=3094.0, deltas=1, output_tokens=97)
    assert burst.burst is True
    assert burst.gen_tok_s is None
    assert burst.ttft_ms == pytest.approx(3094.0)  # the first token's wait is still real


def test_a_normal_stream_keeps_its_generation_rate() -> None:
    stream = _stream_result(latency_ms=5295.0, ttft_ms=2895.0, deltas=40, output_tokens=98)
    assert stream.burst is False
    assert stream.gen_tok_s == pytest.approx(98 / 2.4)


def test_a_stream_that_flushes_few_deltas_is_a_burst_however_long_the_window() -> None:
    """`latency - ttft` of two seconds over three deltas is one flush, timed slowly."""
    flushed = _stream_result(latency_ms=2000.0, ttft_ms=300.0, deltas=3, output_tokens=50)
    spread = _stream_result(latency_ms=2000.0, ttft_ms=300.0, deltas=4, output_tokens=50)
    assert flushed.burst is True  # the first delta plus two: not a generation
    assert spread.burst is False  # the first delta plus three, which is the line


def test_a_refused_stream_is_an_error_not_a_burst() -> None:
    """A 4xx carries no deltas at all, and `post_stream` builds it with empty stats.

    Calling that a burst would put "delivered in one flush" next to an error count, on a
    generation the endpoint refused to start.
    """
    stats = StreamStats()
    stats.on_done(0.12)  # what post_stream does for an error status
    refused = StreamResult(
        message=ParsedMessage(error='{"type": "not_found_error"}'),
        stats=stats,
        status=404,
        latency_ms=120.0,
        started=0.0,
    )
    assert refused.burst is False
    assert refused.gen_tok_s is None  # no rate either way, and nothing to explain away

    records = [
        _record("cold", 0),
        _record("warm", 1, cached=90),
        _record("stream", 0, status=404, error='{"type": "not_found_error"}'),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert not any("burst" in note for note in summary.notes)


def test_a_burst_record_is_noted_and_the_summary_names_its_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record carries `burst`; the spec's notes name the turn by its prompt size."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            return _stream_response(cached=40)
        return _ok(cached=40, input_tokens=60)

    run = _run_probe(
        monkeypatch, [_spec()], _trace(), ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0), handler
    )
    [stream] = [record for record in run.records["fake:model-a"] if record.role == "stream"]
    assert stream.note == BURST_NOTE
    assert stream.gen_tok_s is None

    [summary] = [summarize_probe("fake:model-a", run.records["fake:model-a"])]
    # the note says `-`, the table's own word for "not measured", not "the cell is blank",
    # and names the turn by the prompt size the per-turn table shows, not by its rung number
    burst_note = "sent the 100-token turn's answer in one burst, so tok/s for that turn is -."
    assert burst_note in summary.notes
    rendered = render_probe([summary])
    assert f"fake:model-a: {burst_note}" in rendered
    rung_table = rendered.split("\n\n")[1]  # no caption here: the spec table comes first
    header = [cell.strip() for cell in rung_table.splitlines()[0].split("|")]
    row = [cell.strip() for cell in rung_table.splitlines()[2].split("|")]
    assert row[header.index("tok/s")] == "-"  # the rate is undefined, not zero


def test_a_burst_on_several_turns_names_every_turn_by_its_size() -> None:
    """Two bursting turns read as the sizes the per-turn table shows, not as rung numbers."""
    records = [
        _record("cold", 0, rung=1, prompt=100),
        _record("warm", 1, rung=1, cached=90, prompt=100),
        _record("stream", 0, rung=1).model_copy(update={"note": BURST_NOTE}),
        _record("cold", 0, rung=2, prompt=2_000),
        _record("warm", 1, rung=2, cached=1_800, prompt=2_000),
        _record("stream", 0, rung=2).model_copy(update={"note": BURST_NOTE}),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert (
        "sent its whole answer in one burst on the 100 and 2k turns, so tok/s for those "
        "turns is -." in summary.notes
    )


def test_the_summarys_rate_is_the_median_of_its_timed_rungs() -> None:
    """A burst rung is no rate at all, so it does not pull the median toward zero."""
    records = [
        _record("cold", 0, rung=1),
        _record("warm", 1, rung=1, cached=90),
        _record("cold", 0, rung=2, prompt=200),
        _record("warm", 1, rung=2, cached=190, prompt=200),
    ]
    timed = _record("stream", 0, rung=1)
    timed.gen_tok_s = 40.0
    burst = _record("stream", 0, rung=2, prompt=200)
    burst.note = BURST_NOTE
    summary = summarize_probe("fake:model-a", [*records, timed, burst])
    assert summary.gen_tok_s == 40.0
    assert [rung.gen_tok_s for rung in summary.rungs] == [40.0, None]


# --- the spec column ----------------------------------------------------------

_ROLE_ORDER: dict[str, int] = {"cold": 0, "warm": 1, "stream": 2}


def _summary(
    label: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    prompt: int = 100,
    drift: str | None = None,
    fingerprint: str = "tok1 tok2",
) -> ProbeSummary:
    """One spec's summary with a cold write, a warm hit and a streamed request."""
    records = [
        _record("cold", 0, provider=provider, prompt=prompt),
        _record("warm", 1, cached=90, provider=provider, prompt=prompt),
        ProbeResult(
            spec_label=label,
            rung=1,
            role="stream",
            attempt=0,
            seq=1,
            status=200,
            latency_ms=10.0,
            ttft_ms=100.0,
            gen_tok_s=20.0,
            fingerprint=fingerprint,
            provider=provider,
            model=model,
            prompt_total=prompt,
            cached=90,
        ),
    ]
    summary = summarize_probe(label, records)
    summary.drift = drift
    return summary


def _column_labels(
    summaries: Sequence[ProbeSummary], *, label_width: int = 40
) -> tuple[str | None, list[str]]:
    """The label column of a run's summaries; 40 is the widest cell a table renders."""
    return column_labels([summary.label for summary in summaries], label_width=label_width)


def test_column_labels_shorten_the_openrouter_pair_under_a_caption() -> None:
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash@novita"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@gmicloud"),
    ]
    caption, labels = _column_labels(summaries)
    assert caption == "endpoints: openrouter:deepseek/deepseek-v4.1-flash@<provider>"
    assert labels == ["@novita", "@gmicloud"]


def test_column_labels_keep_a_native_spec_named_and_shorten_the_rest() -> None:
    """The live shape: one native endpoint next to two resellers of the same model."""
    summaries = [
        _summary("deepseek:deepseek-flash"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@novita"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@gmicloud"),
    ]
    caption, labels = _column_labels(summaries)
    assert caption == "endpoints: openrouter:deepseek/deepseek-v4.1-flash@<provider>"
    assert labels == ["deepseek:deepseek-flash", "@novita", "@gmicloud"]
    assert len(set(labels)) == 3


def test_column_labels_never_clip_two_specs_into_one_string() -> None:
    """Two different models of one target are a long, shared prefix: the tail must survive."""
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash-0731"),
    ]
    caption, labels = _column_labels(summaries)
    assert caption is None  # nothing shared, so nothing moves up
    assert labels[0] != labels[1]
    assert len(set(labels)) == 2


def test_column_labels_elide_a_long_label_keeping_its_provider() -> None:
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash-0731@novita"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash-0731@gmicloud"),
    ]
    caption, labels = _column_labels(summaries)
    assert caption is not None
    assert labels == ["@novita", "@gmicloud"]
    assert all(len(label) <= 40 for label in labels)

    # A lone row too long for the width folds its head into the caption instead of eliding:
    # either way the `@provider` that names the row survives, and nothing is cut.
    head = "openrouter:a-very-long-target-name/and-a-long-model"
    lone_caption, lone = _column_labels([_summary(f"{head}@novita")])
    assert lone_caption == f"endpoints: {head}@<provider>"
    assert lone == ["@novita"]


def test_column_labels_folds_a_lone_row_too_long_for_the_width() -> None:
    caption, labels = column_labels(
        ["openrouter:z-ai/glm-5.3-flash@sail-research/fp8"], label_width=32
    )
    assert caption == "endpoints: openrouter:z-ai/glm-5.3-flash@<provider>"
    assert labels == ["@sail-research/fp8"]


def test_column_labels_leaves_a_lone_row_alone_when_it_fits_or_has_no_provider() -> None:
    """The two cases the lone-row fold must not take: a label that already fits keeps its
    whole name, and a native label has no `@provider` tail to leave behind in the column."""
    assert column_labels(["or:model@novita"], label_width=32) == (None, ["or:model@novita"])
    caption, labels = column_labels(
        ["deepseek:deepseek-chat-v3.1-baseline-may-2026"], label_width=32
    )
    assert caption is None
    assert labels == ["deepseek:…v3.1-baseline-may-2026"]


def test_column_labels_does_not_fold_two_differently_headed_rows_even_when_long() -> None:
    """Two or more rows keep today's rule: a head moves into the caption only when at least
    two rows share it, even when each row's own label does not fit the width alone."""
    caption, labels = column_labels(
        [
            "openrouter:z-ai/glm-5.3-flash@sail-research/fp8",
            "openrouter:another-model/name@another-provider/tag",
        ],
        label_width=32,
    )
    assert caption is None
    assert len(set(labels)) == 2


def test_label_floor_of_32_keeps_provider_tags_whole_in_probe_blocks() -> None:
    summaries = [
        _summary("openrouter:z-ai/glm-5.3-flash@sail-research/us"),
        _summary("openrouter:z-ai/glm-5.3-flash@sail-research/fp8"),
    ]
    blocks = probe_blocks(summaries)
    labels = [row[0] for row in blocks.endpoint.rows]
    assert labels == ["@sail-research/us", "@sail-research/fp8"]


def test_probe_blocks_folds_a_single_endpoints_head_when_too_long_for_the_floor() -> None:
    """A one-endpoint probe still gets its `target:model` folded into the caption when its
    own label does not fit -- previously only a shared head across two rows moved up."""
    summaries = [_summary("openrouter:z-ai/glm-5.3-flash@sail-research/fp8")]
    blocks = probe_blocks(summaries)
    assert blocks.caption == "endpoints: openrouter:z-ai/glm-5.3-flash@<provider>"
    assert [row[0] for row in blocks.endpoint.rows] == ["@sail-research/fp8"]


def test_render_probe_uses_the_same_labels_in_both_tables() -> None:
    summaries = [
        _summary("or:model@novita"),
        _summary("or:model@gmicloud", prompt=200),
    ]
    tables = [block.splitlines() for block in render_probe(summaries).split("\n\n")[-2:]]
    spec_table, rung_table = tables[0], tables[1]
    spec_labels = [line.split("|")[0].strip() for line in spec_table[2:]]
    rung_labels = [line.split("|")[0].strip() for line in rung_table[2:]]
    assert spec_labels == ["@novita", "@gmicloud"]
    assert rung_labels == spec_labels


def test_render_probe_notes_use_the_tables_short_label() -> None:
    """Item 16: a per-endpoint note is prefixed with the short column label (`@novita`), not
    the full label the caption above it already expands."""
    summaries = [_summary("or:model@novita"), _summary("or:model@gmicloud")]
    summaries[0].notes = ["a per-endpoint note"]
    rendered = render_probe(summaries)
    assert "@novita: a per-endpoint note" in rendered
    assert "or:model@novita: a per-endpoint note" not in rendered


def test_probe_markdown_notes_use_the_tables_short_label_too() -> None:
    """Item 15/16 apply to the Markdown rendering, which shares `probe_note_lines`."""
    summaries = [_summary("or:model@novita"), _summary("or:model@gmicloud")]
    summaries[1].notes = ["a per-endpoint note"]
    rendered = probe_markdown(summaries)
    assert "@gmicloud: a per-endpoint note" in rendered


def test_render_probe_prints_the_run_wide_cache_mode_note_once() -> None:
    """Item 17: `cold (nonce)` is a fact about the run, not about one endpoint; it prints
    once, unprefixed, as the first note line, in the HTML caveat's own words -- not once per
    endpoint, and not folded into a per-endpoint `label: note` line."""
    summaries = [_summary("or:model@novita"), _summary("or:model@gmicloud")]
    for summary in summaries:
        summary.notes = [cache_mode_note(False), "a per-endpoint note"]
    note_lines = render_probe(summaries).split("\n\n")[-1].splitlines()
    assert note_lines == [
        cache_mode_sentence(False),
        "@novita: a per-endpoint note",
        "@gmicloud: a per-endpoint note",
    ]


def test_render_probe_keeps_a_lone_warm_note_unprefixed_too() -> None:
    """A warm run's short note is just as run-wide as the cold one; it gets the same
    treatment even though it has no elaborated sentence of its own."""
    summaries = [_summary("or:model@novita"), _summary("or:model@gmicloud")]
    for summary in summaries:
        summary.notes = [cache_mode_note(True)]
    note_lines = render_probe(summaries).split("\n\n")[-1].splitlines()
    assert note_lines == [cache_mode_sentence(True)]


def test_session_footer_lines_wraps_into_a_header_and_aligned_rows() -> None:
    """Item 18: the closing `prompt bill` line was one 274-column line at four endpoints; it
    is now a header plus one row per endpoint, named and aligned with the endpoint table's
    own short label."""
    native = _summary("deepseek:deepseek-flash")
    native.session_prompt_usd = 0.0056
    relace = _summary("openrouter:deepseek/deepseek-v4.1-flash@relace/fp4")
    relace.session_prompt_usd = 0.0510
    novita = _summary("openrouter:deepseek/deepseek-v4.1-flash@novita")
    summaries = [native, relace, novita]
    labels = probe_labels(summaries)
    assert labels[0] == "deepseek:deepseek-flash"
    assert labels[1] == "@relace/fp4"
    lines = session_footer_lines(summaries, labels, 1_800_000)
    assert lines == [
        "prompt bill for a session like this trace (1.8 M prompt tokens):",
        "  deepseek:deepseek-flash  $0.0056",
        "  @relace/fp4              $0.0510",
    ]


def test_session_footer_lines_is_empty_without_a_trace_prompt_token_total() -> None:
    summaries = [_summary("fake:model-a")]
    assert session_footer_lines(summaries, probe_labels(summaries), None) == []


def test_render_probe_fits_120_columns_with_every_cell_populated() -> None:
    """The columns between the label and drift never truncate a value, and the table stays
    inside the relaxed plan. Both the label (O2) and the drift column (item 8) are allowed to
    push a whole table past 120 columns on purpose, so the budget is 120 plus the label floor
    (`_LABEL_FLOOR`), the widest a label can add once the spare width falls under it; past
    that the plan has stopped meaning anything."""
    summaries = [
        _summary(
            "deepseek:deepseek-v4.1-flash",
            provider="DeepSeek",
            model="deepseek-chat",
            prompt=20198,
            drift="provider,model,tokens+123%,fingerprint",
        ),
        _summary(
            "openrouter:deepseek/deepseek-v4.1-flash@novita",
            provider="Novita",
            model="deepseek/deepseek-v4.1-flash",
            prompt=20198,
            drift="tokens+123%,fingerprint",
        ),
        _summary(
            "openrouter:deepseek/deepseek-v4.1-flash@gmicloud",
            provider="GMICloud",
            model="deepseek/deepseek-v4.1-flash",
            prompt=20198,
            drift="provider,model,tokens+123%,fingerprint",
        ),
    ]
    budget = 120 + 32  # `probe_tables._MAX_TABLE` plus `_LABEL_FLOOR`, the widest a label adds
    blocks = render_probe(summaries).split("\n\n")
    spec_block, rung_block = blocks[-2], blocks[-1]
    for line in spec_block.splitlines():
        middle = line.split("|", 1)[1].rsplit("|", 1)[0]
        assert "…" not in middle, line
        assert len(line.rsplit("|", 1)[0]) <= budget, line
    for line in rung_block.splitlines():
        middle = line.split("|", 1)[1]
        assert "…" not in middle, line
        assert len(line) <= budget, line


def _thinking_provider(sent: list[bytes]) -> Callable[[httpx.Request], httpx.Response]:
    """A provider that rejects `thinking` only when the output budget is a real one.

    `max_tokens: 1` reads (the cold and warm requests) pass with the parameter; the streamed
    request's 256 meets the reasoning budget and is refused, which is what the retry is for.
    """
    refusal = {"error": {"message": "max_tokens must be greater than thinking.budget_tokens"}}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        body = json.loads(request.content)
        if "thinking" in body and body.get("max_tokens", 1) > 1:
            return httpx.Response(400, json=refusal)
        if body.get("stream"):
            return _stream_response()
        return _ok(cached=40)

    return handler


def test_stream_retries_a_400_about_thinking_without_the_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[bytes] = []
    run = _run_probe(
        monkeypatch,
        [_spec()],
        [_entry(1, thinking=True), _entry(2, thinking=True)],
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0),
        _thinking_provider(sent),
    )

    [stream] = [record for record in run.records["fake:model-a"] if record.role == "stream"]
    assert stream.status == 200
    # a mock transport delivers the whole body at once, so this stream is also a burst
    assert "thinking param dropped" in (stream.note or "").split("; ")
    assert stream.fingerprint == " ".join(_STREAM_TOKENS[:16])
    # the streamed attempt kept the parameter, the retry dropped it
    streamed = [json.loads(body) for body in sent if json.loads(body).get("stream")]
    assert ["thinking" in body for body in streamed] == [True, False]


def test_stream_records_the_error_text_of_a_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(500, json={"error": {"message": "no stream for you"}})
        return _ok(cached=40)

    run = _run_probe(
        monkeypatch,
        [_spec()],
        _trace(),
        ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0),
        handler,
    )
    [stream] = [record for record in run.records["fake:model-a"] if record.role == "stream"]
    assert stream.status == 500
    assert stream.error is not None and "no stream for you" in stream.error
    assert is_failed(stream)


# --- drift --------------------------------------------------------------------


def _drift_summary(label: str, models: Sequence[str], *, prompt: int = 100) -> ProbeSummary:
    """One spec whose served responses named `models`, with its largest rung at `prompt`."""
    records = [_record("cold", 0, prompt=prompt, model=models[0])]
    records.extend(
        _record("warm", index + 1, cached=90, prompt=prompt, model=model)
        for index, model in enumerate(models)
    )
    return summarize_probe(label, records)


def test_model_drift_marks_a_spec_whose_responses_named_two_models() -> None:
    summaries = [
        _drift_summary("or:model@novita", ["model"]),
        _drift_summary("or:model@gmicloud", ["model", "other-model"]),
    ]
    apply_drift(summaries, [_ref("openrouter"), _ref("openrouter")])
    assert summaries[0].drift is None  # the reference is never compared with itself
    assert summaries[1].drift == "model"
    assert summaries[1].models_seen == ["model", "other-model"]
    assert summaries[1].model_seen == "model"  # the first value is the reported one


def test_model_drift_marks_a_model_the_reference_did_not_name() -> None:
    summaries = [
        _drift_summary("or:model@novita", ["model"]),
        _drift_summary("or:model@gmicloud", ["another"]),
    ]
    apply_drift(summaries, [_ref("openrouter"), _ref("openrouter")])
    assert summaries[1].drift == "model"


def test_model_drift_does_not_fire_between_a_native_and_a_gateway_spec() -> None:
    """Each answers with its own name for the same weights: the slug is not drift."""
    summaries = [
        _drift_summary("deepseek:deepseek-flash", ["deepseek-flash"]),
        _drift_summary("or:model@novita", ["deepseek/deepseek-v4.1-flash"]),
    ]
    apply_drift(summaries, [_ref("anthropic"), _ref("openrouter")])
    assert summaries[1].drift is None


def test_model_drift_still_fires_on_the_gateway_spec_that_varies_within_itself() -> None:
    """Across kinds only a spec's own variance is a claim; it is still a claim."""
    summaries = [
        _drift_summary("deepseek:deepseek-flash", ["deepseek-flash"]),
        _drift_summary("or:model@novita", ["deepseek/deepseek-v4.1-flash", "another"]),
    ]
    apply_drift(summaries, [_ref("anthropic"), _ref("openrouter")])
    assert summaries[1].drift == "model"


def test_a_pin_with_a_variant_suffix_is_not_provider_drift_when_that_provider_served() -> None:
    """`@relace/fp4` pins the provider `relace` and the variant `fp4`; a response names only
    `Relace`, so the variant cannot count as drift (G1 of the second reader pass)."""
    reference = _summary("deepseek:deepseek-flash", provider="DeepSeek")
    pinned = _summary("or:model@relace/fp4", provider="Relace")
    elsewhere = _summary("or:model@deepinfra/fp8", provider="Somebody")
    native, relace, deepinfra = _ref("deepseek"), _ref("openrouter"), _ref("openrouter")
    relace.providers = ["relace/fp4"]
    deepinfra.providers = ["deepinfra/fp8"]
    apply_drift([reference, pinned, elsewhere], [native, relace, deepinfra])
    assert pinned.drift is None
    assert elsewhere.drift == "provider"


def test_a_single_model_named_by_both_specs_is_not_drift() -> None:
    summaries = [
        _drift_summary("or:model@novita", ["model"]),
        _drift_summary("or:model@gmicloud", ["model"]),
    ]
    apply_drift(summaries, [_ref("openrouter"), _ref("openrouter")])
    assert summaries[1].drift is None


def test_column_labels_keep_two_long_labels_of_one_target_apart() -> None:
    """Two models of one target differ in the middle of a long label, not at its end."""
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash-0731"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash"),
    ]
    caption, labels = _column_labels(summaries, label_width=23)
    assert caption is None  # different models: nothing moves into a caption
    assert len(set(labels)) == 2
    assert labels[0].endswith("0731")


def test_render_probe_names_every_row_when_two_labels_are_elided() -> None:
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash-0731"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash"),
    ]
    spec_table = render_probe(summaries).split("\n\n")[0].splitlines()
    labels = [line.split("|")[0].strip() for line in spec_table[2:]]
    assert len(set(labels)) == 2


def _ttl_cell(records: list[ProbeResult]) -> str:
    """The `ttl` cell of the rendered rung table, which is what a reader sees."""
    rung_table = render_probe([summarize_probe("fake:model-a", records)]).split("\n\n")[1]
    lines = rung_table.splitlines()
    header = [cell.strip() for cell in lines[0].split("|")]
    row = [cell.strip() for cell in lines[2].split("|")]
    return row[header.index("ttl")]


def _ttl_records(*offsets: int) -> list[ProbeResult]:
    return [
        _record("cold", 0, prompt=100),
        _record("warm", 1, cached=90),
        *(_record("ttl", offset, cached=90) for offset in offsets),
    ]


def test_ttl_cell_marks_a_read_that_went_out_late() -> None:
    """`60s:1 (+6)` says the hit was measured six seconds past the offset it asked for."""
    records = _ttl_records(60, 300)
    [first, second] = [record for record in records if record.role == "ttl"]
    first.offset_actual_s = 61.0  # a second late is inside the margin
    second.offset_actual_s = 306.4  # six seconds late is worth saying
    assert _ttl_cell(records) == "60s:1 300s:1 (+6)"


def test_ttl_cell_says_nothing_when_the_read_was_on_time() -> None:
    records = _ttl_records(60)
    [read] = [record for record in records if record.role == "ttl"]
    read.offset_actual_s = 60.5
    assert _ttl_cell(records) == "60s:1"


def test_ttl_cell_has_no_lateness_without_a_measured_offset() -> None:
    """A record from a run that predates `offset_actual_s` still renders its offset."""
    records = _ttl_records(60)
    [rung] = summarize_probe("fake:model-a", records).rungs
    assert rung.ttl[0].offset_actual_s is None
    assert _ttl_cell(records) == "60s:1"


def test_fingerprint_is_reported_but_never_a_drift_marker() -> None:
    """Two quantizations open differently at temperature 0; that is not a warning."""
    summaries = [
        _summary("or:model@novita"),
        _summary("or:model@gmicloud", fingerprint="Let me read the test files"),
    ]
    apply_drift(summaries, [_ref("openrouter"), _ref("openrouter")])
    assert summaries[1].fingerprint_match is False  # reported
    assert summaries[1].drift is None  # but not a drift marker


def test_a_label_the_caption_does_not_cover_wins_its_width() -> None:
    """The live shape: fifteen `@tag` rows, and the native reference keeps its name whole.

    The `hits` cells are six readings wide, as the default `--repeats` makes them, which is
    what squeezes the label column to its floor when the budget alone decides. O2: the
    reference row's own required width is no longer capped at what the spec table can
    spare the drift floor, so it wins its full name here even though that runs the table
    past `_MAX_TABLE` -- the same trade item 8 already made for the drift column.
    """
    tags = [
        "relace/fp8",
        "deepinfra/turbo",
        "gmicloud",
        "novita/fp8",
        "novita",
        "siliconflow",
        "parasail",
        "friendli",
        "together",
        "fireworks",
        "lambda",
        "cloudflare",
        "baseten",
        "mancer",
        "openinference",
    ]
    summaries = [_summary(f"openrouter:deepseek/deepseek-v4.1-flash@{tag}") for tag in tags]
    summaries.append(_summary("deepseek:deepseek-flash"))
    for summary in summaries:
        summary.rungs[0].hits = [0.9] * 6

    endpoint_block = next(
        block for block in render_probe(summaries).split("\n\n") if block.startswith("endpoint ")
    )
    labels = [line.split("|")[0].strip() for line in endpoint_block.splitlines()[2:]]
    assert labels[-1] == "deepseek:deepseek-flash"  # whole, not folded
    assert len(set(labels)) == len(labels)  # and no two `@tag` rows were cut together


def test_a_label_the_caption_does_not_cover_keeps_its_target_head() -> None:
    """`deepseek:…-chat-v3.1` names a native endpoint; `deeps…-chat-v3.1` reads as a
    reseller of the model the caption names, which is the one thing the row must not say."""
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash@novita"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@gmicloud"),
        _summary("deepseek:deepseek-chat-v3.1"),
    ]
    _, labels = _column_labels(summaries, label_width=17)
    assert labels[2] == "deepseek:…at-v3.1"  # the head whole, the model's tail for the rest

    # A name long enough to still need eliding at the label floor of 32: short enough to fit
    # under it, like `deepseek-chat-v3.1` above, would render whole and prove nothing here.
    long_summaries = [*summaries[:2], _summary("deepseek:deepseek-chat-v3.1-baseline-may-2026")]
    endpoint_block = next(
        block
        for block in render_probe(long_summaries).split("\n\n")
        if block.startswith("endpoint ")
    )
    native = next(line for line in endpoint_block.splitlines() if line.startswith("deepseek"))
    assert native.startswith("deepseek:") and "…" in native


def test_the_label_column_spends_its_columns_and_the_drift_gets_the_rest() -> None:
    """The caption leaves `@provider` tails, so the label column renders narrow, and the
    drift marker `provider,tokens+3%` is read whole regardless of what that costs."""
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash@novita", drift="provider,tokens+3%"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@gmicloud", drift="provider,tokens+3%"),
    ]
    endpoint_block = next(
        block for block in render_probe(summaries).split("\n\n") if block.startswith("endpoint ")
    )
    assert "provider,tokens+3%" in endpoint_block


def test_a_squeezed_table_still_renders_the_whole_drift_marker() -> None:
    """Item 8: numbers wide enough to eat the column budget must not fold the drift marker.

    Before this fix, a squeezed table cut `provider,tokens+3%` down to `provider…`, which
    told a reader something drifted without saying what. The drift column now folds instead
    of truncating, so the whole marker is always on the line even when every other column is
    at its widest.
    """
    summaries = [
        _summary("or:model@novita", drift="provider,tokens+3%"),
        _summary("or:model@gmicloud", drift="provider,tokens+3%"),
        _summary("deepseek:deepseek-flash"),
    ]
    for summary in summaries:
        summary.errors = 12345
        summary.eff_per_m_prompt = 1234.567
        summary.input_price = 1234.567
        rung = summary.rungs[0]
        rung.cold_ms, rung.warm_ms = 12_345_678.0, 87_654_321.0
        rung.ttft_ms, rung.gen_tok_s = 11_111_111.0, 12_345.6

    endpoint_block = next(
        block for block in render_probe(summaries).split("\n\n") if block.startswith("endpoint ")
    )
    cells = [line.split("|")[-1].strip() for line in endpoint_block.splitlines()[2:]]
    assert cells == ["provider,tokens+3%", "provider,tokens+3%", "-"]
    assert all("…" not in cell for cell in cells)


# --- first-read hit rate (item 1) ----------------------------------------------


def test_first_hit_rate_pools_only_each_rungs_first_warm_read() -> None:
    """An agent loop only ever performs a rung's first warm read: later repeats re-read
    what read 1 just wrote, so pooling every repeat flatters the cache the loop never gets
    to rely on. Two rungs, read 1 a miss on one and a hit on the other, read 2+ always a
    hit: `first_hit_rate` must see the 1-of-2 a real agent would, not the pooled 5-of-6."""
    records = [
        _record("cold", 0, rung=1, prompt=100),
        _record("warm", 1, rung=1, cached=0),  # read 1: miss
        _record("warm", 2, rung=1, cached=100),
        _record("warm", 3, rung=1, cached=100),
        _record("cold", 0, rung=2, prompt=100),
        _record("warm", 1, rung=2, cached=100),  # read 1: hit
        _record("warm", 2, rung=2, cached=100),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert summary.hit_rate == pytest.approx(4 / 5)  # 4 hits over 5 served warm reads, pooled
    assert summary.first_hit_rate == pytest.approx(1 / 2)
    assert summary.first_hit_rate is not None
    assert summary.first_cached_fraction is not None
    assert summary.first_h == pytest.approx(summary.first_hit_rate * summary.first_cached_fraction)


def test_first_hit_rate_equals_hit_rate_with_one_repeat() -> None:
    """The acceptance line: `--repeats 1` sends exactly one warm read per rung, so pooling
    every read and pooling only the first read are the same computation."""
    records = [
        _record("cold", 0, rung=1, prompt=100),
        _record("warm", 1, rung=1, cached=100),
        _record("cold", 0, rung=2, prompt=100),
        _record("warm", 1, rung=2, cached=0),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert summary.first_hit_rate == summary.hit_rate
    assert summary.first_cached_fraction == summary.cached_fraction
    assert summary.first_h == summary.h


# --- rate limits (item 6) -------------------------------------------------------


def test_rate_limited_requests_are_told_apart_from_other_errors() -> None:
    """A 429 is the run's own pace, not a refusal; it must count in `errors` (nothing about
    it makes the read usable) but also be named on its own, `rate_limited`, with a note."""
    records = [
        _record("cold", 0, prompt=100),
        _record("warm", 1, status=429, error="rate limited"),
        _record("warm", 2, status=500, error="boom"),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert summary.errors == 2
    assert summary.rate_limited == 1
    assert "1 x 429 rate limit" in summary.notes


def test_no_rate_limit_note_when_nothing_was_rate_limited() -> None:
    records = [_record("cold", 0, prompt=100), _record("warm", 1, cached=100)]
    summary = summarize_probe("fake:model-a", records)
    assert summary.rate_limited == 0
    assert not [note for note in summary.notes if "429" in note]


# --- output tokens (item 4) ------------------------------------------------------


def test_output_tokens_are_summed_and_noted_past_the_max_tokens_1_budget() -> None:
    """GMICloud and native DeepSeek return many output tokens despite `max_tokens: 1`; the
    surplus is worth billing and worth a note, not silently absorbed into the prompt count."""
    cold = _record("cold", 0, prompt=100)
    cold.usage = Usage(output_tokens=200)
    warm = _record("warm", 1, cached=100)
    warm.usage = Usage(output_tokens=150)
    summary = summarize_probe("fake:model-a", [cold, warm])
    assert summary.output_tokens == 350
    assert (
        "ignored the one-token limit on the cache probes and generated 350 tokens. That "
        "raised this run's cost, not the prices above." in summary.notes
    )


def test_no_output_tokens_note_within_the_max_tokens_1_budget() -> None:
    cold = _record("cold", 0, prompt=100)
    cold.usage = Usage(output_tokens=1)
    summary = summarize_probe("fake:model-a", [cold])
    assert summary.output_tokens == 1
    assert not [note for note in summary.notes if "output tokens" in note]


# --- session projection (item 9) -------------------------------------------------


def test_session_prompt_usd_projects_the_measured_price_over_the_trace() -> None:
    """`session_prompt_usd` is the effective price times the trace's own total prompt
    tokens: what a session shaped like the recorded trace would bill in prompt tokens."""
    records = [_record("cold", 0, prompt=100), _record("warm", 1, cached=50)]
    summary = summarize_probe(
        "fake:model-a", records, prices=_prices(), trace_prompt_tokens=2_000_000
    )
    assert summary.eff_per_m_prompt is not None
    assert summary.session_prompt_usd == pytest.approx(summary.eff_per_m_prompt * 2.0)


def test_session_prompt_usd_is_none_without_a_trace_prompt_token_total() -> None:
    records = [_record("cold", 0, prompt=100), _record("warm", 1, cached=50)]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    assert summary.session_prompt_usd is None


# --- no token drift on an incomplete rung set (item 7) --------------------------


def _load_fixture(name: str) -> list[ProbeResult]:
    path = Path(__file__).parent / "fixtures" / name
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ProbeResult.model_validate_json(line) for line in lines if line.strip()]


def test_a_skipped_largest_rung_suppresses_the_spurious_token_drift_marker() -> None:
    """A real case from a tester's run: one spec's largest rung (29) got a 429 on its own
    cold write and was skipped, so its largest *usable* rung (15) is smaller than every
    other spec's. Comparing rung 15's prompt size against the reference's rung 29 reads as
    a huge, spurious `tokens-N%` drift -- two different turns, not two tokenizers -- so the
    marker must be suppressed and a note left in its place."""
    reference_records = _load_fixture("probe_skipped_rung_reference.jsonl")
    candidate_records = _load_fixture("probe_skipped_rung_candidate.jsonl")
    reference = summarize_probe("deepseek:deepseek-flash", reference_records)
    candidate = summarize_probe("or:model@deepinfra", candidate_records)
    assert candidate.rungs[-1].skipped  # rung 29's cold write was refused

    summaries = apply_drift([reference, candidate], [_ref("anthropic"), _ref("openrouter")])

    assert summaries[1].tokens_delta_pct is None
    assert "token drift n/a (rung skipped)" in summaries[1].notes
    assert summaries[1].drift is None or "tokens" not in summaries[1].drift


def test_a_late_ttl_read_does_not_shrink_the_label_column() -> None:
    """The lateness marker is worth its columns; the endpoint column keeps enough to name a row."""
    summaries = [
        _summary("or:model@novita", prompt=20000),
        _summary("or:model@gmicloud", prompt=20000),
    ]
    for summary in summaries:
        summary.rungs[0].ttl = [
            TtlRead(offset_s=60, hit=True, offset_actual_s=66.2),
        ]
    blocks = render_probe(summaries).split("\n\n")
    endpoint_block, rung_block = [
        block for block in blocks if block.startswith(("endpoint ", "endpoint |"))
    ]
    labels = [line.split("|")[0].strip() for line in endpoint_block.splitlines()[2:]]
    assert labels == [
        "@novita",
        "@gmicloud",
    ]  # the shared head is in the caption, so it stays whole
    assert all(len(line) <= 130 for line in endpoint_block.splitlines())
    assert "60s:1 (+6)" in rung_block
