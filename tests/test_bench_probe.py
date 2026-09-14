"""Tests for the probe protocol: nonce, rungs, skip, retries, and the aggregation.

Everything here runs against `httpx.MockTransport`; the only boundary replaced is the
network `httpx.AsyncClient` opens, never the probe logic itself.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
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
from provibench.bench.probe_summary import (
    ProbeSummary,
    TtlRead,
    cached_of,
    summarize_probe,
    summarize_rung,
)
from provibench.bench.probe_tables import render_probe
from provibench.bench.rungs import (
    broadcast_repeats,
    parse_int_list,
    select_rungs,
    validate_rungs,
)
from provibench.bench.targets import Prices, RunSpec, Target, parse_run_spec
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage

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
        prices=Prices(input=1.0, cache_read=0.1, cache_write=1.0, output=2.0), source="table"
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
    assert probe_nonce("abc123", 13) == "provibench-probe:abc123:13"


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


def test_run_probe_sends_cold_then_warm_with_one_nonce_per_rung(
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
    by_rung: dict[int, set[str]] = {}
    for body in bodies:
        nonce = body["system"][0]["text"].splitlines()[0]
        by_rung.setdefault(int(nonce.rsplit(":", 1)[1]), set()).add(nonce)
    assert set(by_rung) == {1, 3}
    assert all(len(nonces) == 1 for nonces in by_rung.values())  # one nonce per rung
    assert by_rung[1] != by_rung[3]  # rung 3 cannot read rung 1's write
    assert all(nonce.startswith(f"provibench-probe:{run.run_hex}:") for nonce in by_rung[1])


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
    assert "cached taken from the OpenRouter generation for 2 request(s)" in summary.notes


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
    assert rung.prefix_fraction == pytest.approx((0.98 + 1.0) / 2)  # capped at 1.0 per hit
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
    assert rung.prefix_fraction is None
    assert rung.hits == []
    assert rung.hit_rate == 0.0
    assert rung.warm_ms is None
    assert rung.errors == 1


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
    assert summary.prefix_fraction == pytest.approx((1.0 + 0.5) / 2)
    assert summary.h == pytest.approx(2 / 3 * 0.75)
    assert summary.eff_per_m_prompt == pytest.approx((1 - 0.5) * 1.0 + 0.5 * 0.1)
    assert summary.price_source == "table"
    assert summary.errors == 0
    assert summary.skipped == 0


def test_summarize_probe_reports_a_skipped_rung_and_no_hit_rate_without_warm_reads() -> None:
    records = [_record("cold", 0, rung=1, status=500, error="boom")]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    assert summary.skipped == 1
    assert summary.hit_rate is None  # nothing was served, so there is no rate to report
    assert summary.h is None
    assert summary.eff_per_m_prompt is None


def test_summarize_probe_caps_the_prefix_fraction_and_falls_back_to_native_cached() -> None:
    records = [
        _record("cold", 0, prompt=100),
        # the response reports nothing, but the gateway says the whole prefix was cached
        _record("warm", 1, cached=0, native_cached=120),
    ]
    summary = summarize_probe("fake:model-a", records)
    assert summary.hit_rate == 1.0
    assert summary.prefix_fraction == 1.0  # 120 / 100 is capped
    assert cached_of(records[1]) == 120
    assert "cached taken from the OpenRouter generation for 1 request(s)" in summary.notes


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
    assert render_probe([]).split("\n\n")[0].startswith("spec")
    assert render_probe([]).split("\n\n")[1].startswith("spec | rung")


def test_render_probe_has_a_spec_table_and_a_rung_table_within_120_columns() -> None:
    records = [
        _record("cold", 0, prompt=100),
        _record("warm", 1, cached=0),
        _record("warm", 2, status=503, error="down"),
        _record("warm", 3, cached=100),
    ]
    summary = summarize_probe("fake:model-a", records, prices=_prices())
    blocks = [block.splitlines() for block in render_probe([summary]).split("\n\n")]
    spec_table, rung_table = blocks[0], blocks[1]
    spec_header = [cell.strip() for cell in spec_table[0].split("|")]
    assert spec_header[:3] == ["spec", "hit %", "prefix %"]
    assert spec_table[2].startswith("fake:model-a")
    rung_header = [cell.strip() for cell in rung_table[0].split("|")]
    assert rung_header[:5] == ["spec", "rung", "prompt", "cached cold", "hits"]
    assert "0 x 1" in rung_table[2]  # one cell per warm attempt, `x` for the failed one
    assert all(len(line) <= 120 for block in blocks for line in block)


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
    assert caption == "specs: openrouter:deepseek/deepseek-v4.1-flash@<provider>"
    assert labels == ["@novita", "@gmicloud"]


def test_column_labels_keep_a_native_spec_named_and_shorten_the_rest() -> None:
    """The live shape: one native endpoint next to two resellers of the same model."""
    summaries = [
        _summary("deepseek:deepseek-flash"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@novita"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@gmicloud"),
    ]
    caption, labels = _column_labels(summaries)
    assert caption == "specs: openrouter:deepseek/deepseek-v4.1-flash@<provider>"
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

    lone = _column_labels([_summary("openrouter:a-very-long-target-name/and-a-long-model@novita")])[
        1
    ]
    assert len(lone[0]) <= 40
    assert lone[0].endswith("@novita")  # the part that names the row survives the cut


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


def test_render_probe_fits_120_columns_with_every_cell_populated() -> None:
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
    for block in render_probe(summaries).split("\n\n"):
        assert all(len(line) <= 120 for line in block.splitlines()), block


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
    assert stream.note == "thinking param dropped"
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

    spec_block = next(
        block for block in render_probe(summaries).split("\n\n") if block.startswith("spec ")
    )
    native = next(line for line in spec_block.splitlines() if line.startswith("deepseek"))
    assert native.startswith("deepseek:") and "…" in native


def test_the_label_column_spends_its_columns_and_the_drift_gets_the_rest() -> None:
    """The caption leaves `@provider` tails, so the label column renders narrow: the
    columns it does not use are the drift's, which is why `provider,tokens+3%` is read
    whole rather than cut to `provider…` while the table still has room for it."""
    summaries = [
        _summary("openrouter:deepseek/deepseek-v4.1-flash@novita", drift="provider,tokens+3%"),
        _summary("openrouter:deepseek/deepseek-v4.1-flash@gmicloud", drift="provider,tokens+3%"),
    ]
    spec_block = next(
        block for block in render_probe(summaries).split("\n\n") if block.startswith("spec ")
    )
    assert "provider,tokens+3%" in spec_block
    assert all(len(line) <= 120 for line in spec_block.splitlines())


def test_a_squeezed_table_still_shows_one_whole_drift_marker() -> None:
    """Numbers wide enough to eat the budget squeeze the drift column: its first marker is
    still read whole. `provider…` says which marker was cut; `provide…` says nothing."""
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

    spec_block = next(
        block for block in render_probe(summaries).split("\n\n") if block.startswith("spec ")
    )
    cells = [line.split("|")[-1].strip() for line in spec_block.splitlines()[2:]]
    assert cells == ["provider…", "provider…", "-"]


def test_a_late_ttl_read_does_not_shrink_the_label_column() -> None:
    """The lateness marker is worth its columns; the spec column keeps enough to name a row."""
    summaries = [
        _summary("or:model@novita", prompt=20000),
        _summary("or:model@gmicloud", prompt=20000),
    ]
    for summary in summaries:
        summary.rungs[0].ttl = [
            TtlRead(offset=60, hit=True, offset_actual_s=66.2),
        ]
    blocks = render_probe(summaries).split("\n\n")
    spec_block, rung_block = [block for block in blocks if block.startswith(("spec ", "spec |"))]
    labels = [line.split("|")[0].strip() for line in spec_block.splitlines()[2:]]
    assert labels == [
        "@novita",
        "@gmicloud",
    ]  # the shared head is in the caption, so it stays whole
    assert all(len(line) <= 120 for line in spec_block.splitlines())
    assert "60s:1 (+6)" in rung_block
