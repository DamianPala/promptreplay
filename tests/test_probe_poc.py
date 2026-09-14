"""Tests for the script-only probe PoC: nonce stamping, rung selection, aggregation, CLI."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from provibench.bench.targets import RunSpec, Target
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage, append_entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Pyright cannot follow a computed `sys.path` entry, so the script module is loaded
# dynamically and pinned to `Any` here. It gets its own strict check through
# `uv run pyright scripts/probe_poc.py`.
probe_poc: Any = importlib.import_module("probe_poc")

_RUN_HEX = "0123456789ab"
_NONCE = f"provibench-probe:{_RUN_HEX}:1"
_ENTRY_HEADERS = {"anthropic-version": "2023-06-01"}


def _target(kind: Literal["openrouter", "anthropic"] = "anthropic") -> Target:
    return Target(
        name="fake", url="https://api.test/v1/messages", api_key_env="FAKE_KEY", kind=kind
    )


def _openrouter_spec() -> RunSpec:
    return RunSpec(target=_target("openrouter"), model="deepseek/model", providers=["novita"])


def _spec(
    model: str = "model-a", kind: Literal["openrouter", "anthropic"] = "anthropic"
) -> RunSpec:
    return RunSpec(target=_target(kind), model=model)


def _system() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "You are helpful."},
        {"type": "text", "text": "Repo context.", "cache_control": {"type": "ephemeral"}},
    ]


def _entry(seq: int, pairs: int) -> TraceEntry:
    """A turn whose messages grow with `seq`, sharing one system prompt."""
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
    return TraceEntry(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        path="/v1/messages",
        headers=dict(_ENTRY_HEADERS),
        body=body,
        conversation="c1",
        response=RecordedResponse(
            status=200,
            latency_ms=1.0,
            usage=Usage(input_tokens=10, cache_read_input_tokens=5),
        ),
    )


def _trace() -> list[TraceEntry]:
    return [_entry(seq, seq) for seq in range(1, 5)]


def _state(run_hex: str = _RUN_HEX) -> Any:
    return probe_poc.ProbeState(
        run_hex=run_hex, trace="trace.jsonl", started="2026-01-01T00:00:00Z"
    )


def _config(
    specs: list[Any],
    *,
    rungs: list[int],
    repeats: list[int],
    gap_s: float = 0.0,
) -> Any:
    return probe_poc.ProbeConfig(
        trace=Path("trace.jsonl"), specs=specs, rungs=rungs, repeats=repeats, gap_s=gap_s
    )


def _record(
    role: Literal["cold", "warm"],
    attempt: int,
    *,
    status: int = 200,
    cached: int = 0,
    prompt: int = 100,
    cache_write: int = 0,
    error: str | None = None,
    latency: float = 10.0,
) -> Any:
    return probe_poc.RequestRecord(
        spec="s",
        rung=1,
        nonce=_NONCE,
        role=role,
        attempt=attempt,
        status=status,
        latency_ms=latency,
        prompt_total=prompt,
        input_tokens=max(prompt - cached, 0),
        cached=cached,
        cache_write=cache_write,
        error=error,
    )


# --- nonce stamping -----------------------------------------------------------


def test_build_probe_body_stamps_first_system_block_and_keeps_cache_control() -> None:
    entry = _trace()[0]
    prepared = probe_poc.build_probe_body(
        entry, _spec(), probe_poc.ReplayOptions(max_tokens=1), "N"
    )
    assert prepared["system"][0]["text"] == "N\nYou are helpful."
    assert prepared["system"][1] == {
        "type": "text",
        "text": "Repo context.",
        "cache_control": {"type": "ephemeral"},
    }
    assert prepared["stream"] is False
    assert prepared["max_tokens"] == 1


def test_build_probe_body_leaves_the_recorded_entry_untouched() -> None:
    entry = _trace()[0]
    probe_poc.build_probe_body(entry, _spec(), probe_poc.ReplayOptions(max_tokens=1), "N")
    assert entry.body["system"][0]["text"] == "You are helpful."
    assert entry.body["stream"] is True
    assert entry.body["max_tokens"] == 1024


def test_inject_nonce_prefixes_a_string_system_and_copies_the_body() -> None:
    body: dict[str, Any] = {"system": "hello", "messages": []}
    stamped = probe_poc.inject_nonce(body, "N")
    assert stamped["system"] == "N\nhello"
    assert body["system"] == "hello"


def test_inject_nonce_rejects_a_body_it_cannot_stamp() -> None:
    with pytest.raises(ValueError, match="no 'system'"):
        probe_poc.inject_nonce({"messages": []}, "N")
    with pytest.raises(ValueError, match="no 'system'"):
        probe_poc.inject_nonce({"system": None}, "N")
    with pytest.raises(ValueError, match="non-empty list"):
        probe_poc.inject_nonce({"system": []}, "N")
    with pytest.raises(ValueError, match="non-empty list"):
        probe_poc.inject_nonce({"system": {"type": "text", "text": "x"}}, "N")
    with pytest.raises(ValueError, match="no 'text'"):
        probe_poc.inject_nonce({"system": [{"type": "text"}]}, "N")
    with pytest.raises(ValueError, match="no 'text'"):
        probe_poc.inject_nonce({"system": ["plain"]}, "N")


def test_rung_nonce_derives_from_the_run_hex() -> None:
    assert probe_poc.rung_nonce("abc123", 13) == "provibench-probe:abc123:13"


# --- rung selection -----------------------------------------------------------


def test_parse_int_list_accepts_spaces_and_rejects_empty() -> None:
    assert probe_poc.parse_int_list("1, 13 ,30") == [1, 13, 30]
    with pytest.raises(ValueError, match="comma-separated list"):
        probe_poc.parse_int_list(" , ")
    with pytest.raises(ValueError, match="comma-separated list"):
        probe_poc.parse_int_list("1,x")


def test_broadcast_repeats_repeats_the_last_value_for_missing_entries() -> None:
    assert probe_poc.broadcast_repeats([1, 13, 30], [5, 2]) == [5, 2, 2]


def test_broadcast_repeats_ignores_entries_past_the_last_rung() -> None:
    assert probe_poc.broadcast_repeats([1, 2], [3, 4, 5]) == [3, 4]


def test_validate_rungs_rejects_a_rung_without_a_next_turn() -> None:
    with pytest.raises(ValueError, match="rung 4 needs turn 5"):
        probe_poc.validate_rungs([4], _trace())


def test_validate_rungs_accepts_the_last_valid_rung() -> None:
    probe_poc.validate_rungs([1, 3], _trace())


def test_validate_rungs_rejects_an_empty_trace() -> None:
    with pytest.raises(ValueError, match="no conversations"):
        probe_poc.validate_rungs([1], [])


# --- aggregation --------------------------------------------------------------


def test_summarize_rung_hit_rate_fractions_and_cache_write() -> None:
    records = [
        _record("cold", 0, cached=0, prompt=100, cache_write=50, latency=20.0),
        _record("warm", 1, cached=0, latency=30.0),
        _record("warm", 2, cached=98, latency=40.0),
        _record("warm", 3, cached=200, latency=50.0),
    ]
    aggregate = probe_poc.summarize_rung("s", 1, _NONCE, records)
    assert aggregate.nonce == _NONCE
    assert aggregate.prompt_cold == 100
    assert aggregate.prefix == 100
    assert aggregate.cached_cold == 0
    assert aggregate.hit_rate == pytest.approx(2 / 3)
    assert aggregate.frac == pytest.approx((0.98 + 1.0) / 2)  # capped at 1.0 per hit
    assert aggregate.hits == [0.0, 0.98, 1.0]
    assert aggregate.cache_write_cold == 50
    assert aggregate.ttft_cold_ms == pytest.approx(20.0)
    assert aggregate.ttft_warm_ms == pytest.approx(40.0)  # median over warm attempts
    assert aggregate.skipped is False
    assert aggregate.cold_error is None
    assert aggregate.errors == 0


def test_summarize_rung_ttft_warm_median_ignores_failed_attempts() -> None:
    records = [
        _record("cold", 0),
        _record("warm", 1, status=500, error="boom", latency=5.0),
        _record("warm", 2, cached=10, latency=100.0),
        _record("warm", 3, cached=10, latency=300.0),
    ]
    aggregate = probe_poc.summarize_rung("s", 1, _NONCE, records)
    assert aggregate.ttft_warm_ms == pytest.approx(200.0)  # median of 100, 300 only


def test_summarize_rung_without_hits_has_no_fraction() -> None:
    records = [_record("cold", 0), _record("warm", 1), _record("warm", 2)]
    aggregate = probe_poc.summarize_rung("s", 1, _NONCE, records)
    assert aggregate.hit_rate == 0.0
    assert aggregate.frac is None
    assert aggregate.hits == [0.0, 0.0]


def test_summarize_rung_marks_failed_warm_attempts_with_x() -> None:
    records = [
        _record("cold", 0),
        _record("warm", 1, status=500, error="boom"),
        _record("warm", 2, status=0, error="connect failed"),
        _record("warm", 3, cached=50),
    ]
    aggregate = probe_poc.summarize_rung("s", 1, _NONCE, records)
    assert aggregate.errors == 2
    assert aggregate.hit_rate == 1.0  # only the served attempts are a denominator
    assert aggregate.hits == ["x", "x", 0.5]


def test_summarize_rung_skips_a_rung_whose_cold_request_failed() -> None:
    records = [_record("cold", 0, status=500, error="boom", cache_write=0)]
    aggregate = probe_poc.summarize_rung("s", 1, _NONCE, records)
    assert aggregate.skipped is True
    assert aggregate.cold_error == "boom"
    assert aggregate.frac is None
    assert aggregate.hits == []
    assert aggregate.hit_rate == 0.0
    assert aggregate.ttft_warm_ms is None
    assert aggregate.errors == 1


def test_summarize_rung_with_only_errors_has_no_fraction() -> None:
    records = [_record("cold", 0), _record("warm", 1, status=503, error="down")]
    aggregate = probe_poc.summarize_rung("s", 1, _NONCE, records)
    assert aggregate.hit_rate == 0.0
    assert aggregate.frac is None
    assert aggregate.hits == ["x"]
    assert aggregate.errors == 1


def test_exit_code_is_zero_only_when_every_request_succeeded() -> None:
    ok = [_record("cold", 0), _record("warm", 1)]
    assert probe_poc.exit_code(ok) == 0
    assert probe_poc.exit_code([*ok, _record("warm", 2, status=500, error="boom")]) == 1
    assert probe_poc.exit_code([*ok, _record("warm", 3, status=0, error="down")]) == 1


def test_render_report_has_a_header_row_per_rung_and_a_spec_summary() -> None:
    records = [
        _record("cold", 0, cached=0, prompt=100),
        _record("warm", 1, cached=0, prompt=100),
        _record("warm", 2, status=503, error="down"),
        _record("warm", 3, cached=50, prompt=100),
    ]
    report = probe_poc.render_report([probe_poc.summarize_rung("s", 1, _NONCE, records)])
    lines = report.splitlines()
    assert [cell.strip() for cell in lines[0].split(" | ")] == list(probe_poc._HEADERS)
    assert "cached cold" in lines[0]
    assert "s" in lines[2]
    assert "x" in lines[2]  # the failed attempt renders as `x`
    assert lines[-1] == "s: mean frac 0.50 over 1 rung(s)"


# --- end to end (no network) --------------------------------------------------


def _ok(cached: int = 0, input_tokens: int = 10) -> httpx.Response:
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


def _write_trace(path: Path) -> None:
    for entry in _trace():
        append_entry(path, entry)


_TARGETS_TOML = """
[targets.fake]
url = "https://api.test/v1/messages"
api_key_env = "FAKE_KEY"
kind = "anthropic"
"""


def _write_targets(path: Path) -> None:
    path.write_text(_TARGETS_TOML, encoding="utf-8")


def _run_probe(config: Any, entries: Any, api_keys: dict[str, str], handler: Any = None) -> Any:
    """Run `run_probe` against a mock transport, with `_post` already patched."""
    transport = httpx.MockTransport(handler or (lambda request: httpx.Response(404, json={})))

    async def go() -> Any:
        async with httpx.AsyncClient(transport=transport) as client:
            return await probe_poc.run_probe(
                config, entries, api_keys, state=_state(), client=client
            )

    return asyncio.run(go())


def _fake_post(responses: list[httpx.Response], sent: list[dict[str, Any]] | None = None) -> Any:
    """A `_post` stand-in that pops one queued response per call."""
    calls = {"n": 0}

    async def fake_post(
        client: httpx.AsyncClient,
        spec: RunSpec,
        headers: dict[str, str],
        body: dict[str, Any],
        opts: Any,
    ) -> httpx.Response:
        if sent is not None:
            sent.append(body)
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[index]

    return fake_post


def test_main_writes_the_json_report_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    targets_path = tmp_path / "targets.toml"
    out_path = tmp_path / "out.json"
    _write_trace(trace_path)
    _write_targets(targets_path)
    monkeypatch.setenv("FAKE_KEY", "secret")

    bodies: list[dict[str, Any]] = []
    calls = {"n": 0}

    async def fake_post(
        client: httpx.AsyncClient,
        spec: RunSpec,
        headers: dict[str, str],
        body: dict[str, Any],
        opts: Any,
    ) -> httpx.Response:
        bodies.append(body)
        index = calls["n"]
        calls["n"] += 1
        # cold: 100-token prompt. warm: 90 of those 100 tokens read from cache.
        return _ok(cached=0, input_tokens=100) if index == 0 else _ok(cached=90, input_tokens=10)

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    code = probe_poc.main(
        [
            str(trace_path),
            "fake:model-a",
            "--rungs",
            "1",
            "--repeats",
            "2",
            "--gap",
            "0",
            "--targets",
            str(targets_path),
            "--output-file",
            str(out_path),
        ]
    )

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload) == {"run_hex", "trace", "started", "requests", "rungs"}
    assert payload["trace"] == str(trace_path)
    assert len(payload["run_hex"]) == 12
    nonce = f"provibench-probe:{payload['run_hex']}:1"
    assert [record["role"] for record in payload["requests"]] == ["cold", "warm", "warm"]
    assert [record["attempt"] for record in payload["requests"]] == [0, 1, 2]
    assert all(record["spec"] == "fake:model-a" for record in payload["requests"])
    assert all(record["nonce"] == nonce for record in payload["requests"])
    assert payload["rungs"][0]["nonce"] == nonce
    # cold and warm share the rung's nonce, so the warm prefix equals the cold prompt
    assert [body["system"][0]["text"].splitlines()[0] for body in bodies] == [nonce] * 3
    assert payload["rungs"][0]["hit_rate"] == 1.0
    assert payload["rungs"][0]["frac"] == pytest.approx(0.9)
    assert payload["rungs"][0]["cached_cold"] == 0
    assert payload["rungs"][0]["skipped"] is False


def test_main_exits_one_and_still_writes_output_when_a_request_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    targets_path = tmp_path / "targets.toml"
    out_path = tmp_path / "out.json"
    _write_trace(trace_path)
    _write_targets(targets_path)
    monkeypatch.setenv("FAKE_KEY", "secret")

    responses = [_ok(), httpx.Response(500, json={"error": {"message": "boom"}})]
    monkeypatch.setattr(probe_poc, "_post", _fake_post(responses))

    code = probe_poc.main(
        [
            str(trace_path),
            "fake:model-a",
            "--rungs",
            "1",
            "--repeats",
            "1",
            "--gap",
            "0",
            "--targets",
            str(targets_path),
            "--output-file",
            str(out_path),
        ]
    )

    assert code == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["requests"][1]["status"] == 500
    assert "boom" in (payload["requests"][1]["error"] or "")
    assert payload["rungs"][0]["hits"] == ["x"]
    assert payload["rungs"][0]["errors"] == 1


def test_main_keeps_partial_records_when_the_run_fails_unexpectedly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    targets_path = tmp_path / "targets.toml"
    out_path = tmp_path / "out.json"
    _write_trace(trace_path)
    _write_targets(targets_path)
    monkeypatch.setenv("FAKE_KEY", "secret")

    calls = {"n": 0}

    async def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("interrupted")
        return _ok()

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    with pytest.raises(RuntimeError, match="interrupted"):
        probe_poc.main(
            [
                str(trace_path),
                "fake:model-a",
                "--rungs",
                "1",
                "--repeats",
                "3",
                "--gap",
                "0",
                "--targets",
                str(targets_path),
                "--output-file",
                str(out_path),
            ]
        )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["requests"]) == 2  # the cold request and the first warm attempt
    assert [record["attempt"] for record in payload["requests"]] == [0, 1]


def test_main_rejects_a_rung_without_a_next_turn_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    targets_path = tmp_path / "targets.toml"
    _write_trace(trace_path)
    _write_targets(targets_path)
    monkeypatch.setenv("FAKE_KEY", "secret")

    sent = {"n": 0}

    async def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        sent["n"] += 1
        return _ok()

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    code = probe_poc.main(
        [
            str(trace_path),
            "fake:model-a",
            "--rungs",
            "4",
            "--targets",
            str(targets_path),
            "--gap",
            "0",
        ]
    )

    assert code == 2
    assert sent["n"] == 0
    assert "rung 4 needs turn 5" in capsys.readouterr().err


def test_main_reports_a_missing_api_key_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    targets_path = tmp_path / "targets.toml"
    _write_trace(trace_path)
    _write_targets(targets_path)
    monkeypatch.delenv("FAKE_KEY", raising=False)

    code = probe_poc.main([str(trace_path), "fake:model-a", "--targets", str(targets_path)])

    assert code == 2
    assert "FAKE_KEY" in capsys.readouterr().err


# --- run_probe behaviour ------------------------------------------------------


def test_run_probe_sends_all_rungs_of_a_spec_before_the_next_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    async def fake_post(
        client: httpx.AsyncClient,
        spec: RunSpec,
        headers: dict[str, str],
        body: dict[str, Any],
        opts: Any,
    ) -> httpx.Response:
        labels.append(spec.label)
        return _ok()

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    targets = {"fake": _target()}
    specs = [
        probe_poc.parse_run_spec("fake:a", targets),
        probe_poc.parse_run_spec("fake:b", targets),
    ]
    run = _run_probe(
        _config(specs, rungs=[1, 2], repeats=[1, 1]),
        _trace(),
        {"fake:a": "k", "fake:b": "k"},
    )

    assert labels == ["fake:a"] * 4 + ["fake:b"] * 4  # cold + warm for each of rungs 1, 2
    assert [(aggregate.spec, aggregate.rung) for aggregate in run.rungs] == [
        ("fake:a", 1),
        ("fake:a", 2),
        ("fake:b", 1),
        ("fake:b", 2),
    ]


def test_run_probe_skips_a_rung_whose_cold_request_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    responses = [
        httpx.Response(500, json={"error": {"message": "boom"}}),  # rung 1 cold fails
        _ok(),
        _ok(),
        _ok(),  # rung 3: cold + two warm
    ]
    monkeypatch.setattr(probe_poc, "_post", _fake_post(responses, sent))

    run = _run_probe(
        _config([_spec()], rungs=[1, 3], repeats=[3, 2]), _trace(), {"fake:model-a": "k"}
    )

    # rung 1 sent only its cold request; rung 3 ran in full
    assert [(record.rung, record.role) for record in run.requests] == [
        (1, "cold"),
        (3, "cold"),
        (3, "warm"),
        (3, "warm"),
    ]
    skipped, attempted = run.rungs
    assert skipped.skipped is True
    assert skipped.cold_error is not None and "boom" in skipped.cold_error
    assert skipped.frac is None
    assert skipped.hits == []
    assert skipped.errors == 1
    assert attempted.skipped is False


def test_run_probe_uses_one_nonce_per_rung_shared_across_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []

    async def fake_post(
        client: httpx.AsyncClient,
        spec: RunSpec,
        headers: dict[str, str],
        body: dict[str, Any],
        opts: Any,
    ) -> httpx.Response:
        sent.append((spec.label, body))
        return _ok()

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    targets = {"fake": _target()}
    specs = [
        probe_poc.parse_run_spec("fake:a", targets),
        probe_poc.parse_run_spec("fake:b", targets),
    ]
    run = _run_probe(
        _config(specs, rungs=[1, 3], repeats=[1, 1]),
        _trace(),
        {"fake:a": "k", "fake:b": "k"},
    )

    def nonce_of(body: dict[str, Any]) -> str:
        return body["system"][0]["text"].splitlines()[0]

    by_spec: dict[str, list[str]] = {"fake:a": [], "fake:b": []}
    for label, body in sent:
        by_spec[label].append(nonce_of(body))
    assert len(set(by_spec["fake:a"])) == 2  # rung 1 and rung 3 differ
    assert by_spec["fake:a"] == by_spec["fake:b"]  # but every spec agrees per rung
    assert all(nonce.startswith(f"provibench-probe:{run.run_hex}:") for nonce in by_spec["fake:a"])
    assert [aggregate.nonce for aggregate in run.rungs] == [
        f"provibench-probe:{run.run_hex}:1",
        f"provibench-probe:{run.run_hex}:3",
        f"provibench-probe:{run.run_hex}:1",
        f"provibench-probe:{run.run_hex}:3",
    ]


def test_warm_attempts_send_byte_identical_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    serialized: list[str] = []

    async def fake_post(
        client: httpx.AsyncClient,
        spec: RunSpec,
        headers: dict[str, str],
        body: dict[str, Any],
        opts: Any,
    ) -> httpx.Response:
        serialized.append(json.dumps(body, sort_keys=True))
        return _ok()

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    _run_probe(_config([_spec()], rungs=[1], repeats=[3]), _trace(), {"fake:model-a": "k"})

    cold, *warm = serialized
    assert len(warm) == 3
    assert len(set(warm)) == 1  # the repeats are the same bytes, not the same object
    assert cold != warm[0]  # turn k and turn k+1 are different requests


def test_gap_is_waited_before_every_warm_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(probe_poc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(probe_poc, "_post", _fake_post([_ok()]))

    _run_probe(
        _config([_spec()], rungs=[1], repeats=[3], gap_s=0.5), _trace(), {"fake:model-a": "k"}
    )

    assert sleeps == [0.5, 0.5, 0.5]


def test_send_retries_429_and_503_with_the_backoff_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(probe_poc.asyncio, "sleep", fake_sleep)
    responses = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(503, json={"error": {"message": "overloaded"}}),
        _ok(cached=50),
    ]
    monkeypatch.setattr(probe_poc, "_post", _fake_post(responses))

    run = _run_probe(_config([_spec()], rungs=[1], repeats=[0]), _trace(), {"fake:model-a": "k"})

    assert len(run.requests) == 1
    assert run.requests[0].status == 200
    assert run.requests[0].retries == 2
    assert run.requests[0].cached == 50
    assert sleeps == [2.0, 5.0]


def test_send_does_not_retry_a_500(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    monkeypatch.setattr(probe_poc, "_post", fake_post)

    run = _run_probe(_config([_spec()], rungs=[1], repeats=[0]), _trace(), {"fake:model-a": "k"})

    assert calls["n"] == 1
    assert run.requests[0].retries == 0
    assert run.requests[0].status == 500


# --- OpenRouter generation enrichment -----------------------------------------

_ENDPOINTS = {
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


def _openrouter_handler(*, native_cached: int = 45, generation_fails: bool = False) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=_ENDPOINTS)
        if request.url.path.endswith("/generation"):
            if generation_fails:
                raise httpx.ConnectError("generation lookup down", request=request)
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "gen_1",
                        "provider_name": "Novita",
                        "native_tokens_prompt": 100,
                        "native_tokens_cached": native_cached,
                        "native_tokens_completion": 1,
                    }
                },
            )
        return httpx.Response(404, json={})

    return handler


def _run_openrouter(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response],
    *,
    native_cached: int = 45,
    generation_fails: bool = False,
) -> Any:
    monkeypatch.setattr(probe_poc, "_post", _fake_post(responses))
    spec = _openrouter_spec()
    return _run_probe(
        _config([spec], rungs=[1], repeats=[1]),
        _trace(),
        {spec.label: "k"},
        _openrouter_handler(native_cached=native_cached, generation_fails=generation_fails),
    )


def test_enrichment_fills_provider_and_native_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run_openrouter(monkeypatch, [_ok(cached=40, input_tokens=60)], native_cached=45)

    cold, warm = run.requests
    for record in (cold, warm):
        assert record.provider == "Novita"
        assert record.native_tokens_cached == 45
        assert record.cached == 40  # response usage wins while it reports a hit
        assert record.cached_source == "response"
        assert record.note is None


def test_enrichment_failure_keeps_the_response_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run_openrouter(monkeypatch, [_ok(cached=40, input_tokens=60)], generation_fails=True)

    for record in run.requests:
        assert record.cached == 40
        assert record.prompt_total == 100
        assert record.native_tokens_cached is None
        assert record.note is not None and "generation enrichment failed" in record.note


def test_enrichment_falls_back_to_the_native_cached_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run_openrouter(
        monkeypatch,
        [_ok(cached=0, input_tokens=100), _ok(cached=0, input_tokens=100)],
        native_cached=40,
    )

    for record in run.requests:
        assert record.cached == 40
        assert record.cached_source == "openrouter-generation"
        assert record.native_tokens_cached == 40
    assert run.rungs[0].cached_cold == 40  # the fallback feeds the aggregate too
    assert run.rungs[0].hit_rate == 1.0
    assert run.rungs[0].hits == [0.4]  # the warm attempt hit 40 of the cold's 100
    assert run.rungs[0].skipped is False


def test_enrichment_does_not_fall_back_when_the_response_reports_a_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run_openrouter(
        monkeypatch,
        [_ok(cached=40, input_tokens=60), _ok(cached=40, input_tokens=60)],
        native_cached=999,
    )

    for record in run.requests:
        assert record.cached == 40
        assert record.cached_source == "response"
