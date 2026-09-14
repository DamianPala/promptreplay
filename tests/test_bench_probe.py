"""Tests for the probe protocol: nonce, rungs, skip, retries, and the aggregation.

Everything here runs against `httpx.MockTransport`; the only boundary replaced is the
network `httpx.AsyncClient` opens, never the probe logic itself.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Literal

import httpx
import pytest

from provibench.bench.estimate import SpecPrices
from provibench.bench.nonce import inject_nonce, probe_nonce, require_stampable, run_nonce
from provibench.bench.probe import (
    ProbeOptions,
    ProbeResult,
    ProbeRun,
    is_failed,
    is_served,
    run_probe,
)
from provibench.bench.probe_summary import cached_of, render_probe, summarize_probe, summarize_rung
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
    role: Literal["cold", "warm"],
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
        prompt_total=prompt,
        cached=cached,
        cache_write=cache_write,
        native_tokens_cached=native_cached,
        generation=generation,
        error=error,
    )


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
        monkeypatch, specs, _trace(), ProbeOptions(rungs=[1, 3], repeats=[1, 1], gap_s=0.0), handler
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
        ProbeOptions(rungs=[1, 3], repeats=[3, 2], gap_s=0.0),
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
        monkeypatch, [_spec()], _trace(), ProbeOptions(rungs=[1], repeats=[3], gap_s=0.0), handler
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
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0, warm=True),
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
        ProbeOptions(rungs=[1], repeats=[2], gap_s=0.0),
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
        monkeypatch, [_spec()], _trace(), ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0), handler
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
        monkeypatch, [_spec()], _trace(), ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0), handler
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
        ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0),
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
        ProbeOptions(rungs=[1], repeats=[3], gap_s=0.0),
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
        ProbeOptions(rungs=[1], repeats=[3], gap_s=0.0),
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
        ProbeOptions(rungs=[1], repeats=[0], gap_s=0.0),
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
        monkeypatch, [spec], _trace(), ProbeOptions(rungs=[1], repeats=[1], gap_s=0.0), handler
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
