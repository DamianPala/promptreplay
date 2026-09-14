"""`probe` and `report`: the estimate, the budget, the run it persists, and the report.

The only boundary mocked is the network `httpx.AsyncClient` opens, so these tests exercise
the command, the probe protocol and the aggregation together, on recorded fixtures.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from provibench.bench.trace import (
    RecordedResponse,
    TraceEntry,
    Usage,
    append_entry,
    write_trace_text,
)
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli

_RealAsyncClient = httpx.AsyncClient

_TARGETS_TOML = (
    "[targets.fake]\n"
    'url = "https://api.test/v1/messages"\n'
    'api_key_env = "FAKE_KEY"\n'
    'kind = "anthropic"\n'
    "\n"
    '[targets.fake.prices."model-a"]\n'
    "input = 1.0\n"
    "cache_read = 0.1\n"
    "cache_write = 1.0\n"
    "output = 2.0\n"
    "\n"
    "[targets.or]\n"
    'url = "https://openrouter.test/v1/messages"\n'
    'api_key_env = "OR_KEY"\n'
    'kind = "openrouter"\n'
)

_ENDPOINTS: dict[str, Any] = {
    "data": {
        "endpoints": [
            {
                "provider_name": "Novita",
                "tag": "novita",
                "quantization": "fp8",
                "context_length": 128000,
                "pricing": {
                    "prompt": "0.0000003",
                    "completion": "0.0000006",
                    "input_cache_read": "0.00000003",
                    "input_cache_write": "0.0000003",
                },
            }
        ]
    }
}

_GENERATION: dict[str, Any] = {
    "id": "gen_1",
    "provider_name": "Novita",
    "native_tokens_prompt": 100,
    "native_tokens_cached": 90,
    "native_tokens_completion": 1,
    "total_cost": 0.0004,
}

_ENV = {"FAKE_KEY": "secret", "OR_KEY": "secret"}


def _write_targets(path: Path) -> None:
    path.write_text(_TARGETS_TOML)


def _trace_entry(seq: int, prompt_tokens: int) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        body={
            "model": "orig",
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Repo context.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": f"question {seq}"}],
            "max_tokens": 1024,
        },
        conversation="c1",
        response=RecordedResponse(
            status=200, latency_ms=1.0, usage=Usage(input_tokens=prompt_tokens)
        ),
    )


def _write_trace(path: Path, prompts: tuple[int, ...] = (100, 200, 300)) -> None:
    for seq, prompt_tokens in enumerate(prompts, start=1):
        append_entry(path, _trace_entry(seq, prompt_tokens))


def _ok(cached: int = 0, input_tokens: int = 100, model: str = "model-a") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "model": model,
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


def _delta(index: int, kind: str, text: str) -> dict[str, Any]:
    """One content delta; `kind` is `thinking_delta` or `text_delta`, each with its key."""
    key = "thinking" if kind == "thinking_delta" else "text"
    return {"type": "content_block_delta", "index": index, "delta": {"type": kind, key: f" {text}"}}


def _stream_response(
    body: dict[str, Any] | None = None,
    *,
    cached: int = 90,
    model: str = "model-a",
    input_tokens: int = 10,
) -> httpx.Response:
    """A streamed generation: message_start, thinking deltas, text deltas, usage, stop."""
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
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        *(_delta(0, "thinking_delta", token) for token in _STREAM_TOKENS[:8]),
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        *(_delta(1, "text_delta", token) for token in _STREAM_TOKENS[8:]),
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 18},
        },
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200, content=_sse_bytes(events), headers={"content-type": "text/event-stream"}
    )


def _refuse_stream(body: dict[str, Any] | None = None) -> httpx.Response:
    """A provider that answers the throughput request with a 500."""
    del body
    return httpx.Response(500, json={"error": {"message": "no stream"}})


def _transport(
    reply: Callable[[int], httpx.Response],
    *,
    sent: list[dict[str, Any]] | None = None,
    generation: dict[str, Any] | None = _GENERATION,
    stream: Callable[[dict[str, Any]], httpx.Response] | None = None,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """A transport serving /endpoints and /generation, and delegating the rest to `reply`.

    A request whose body carries `stream: true` goes to `stream` instead of `reply`, so a
    test can give the throughput request its own answer without counting it as a message.
    `handler` replaces the whole message path for a test that needs to see every request.
    """
    calls = {"messages": 0}

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=_ENDPOINTS)
        if request.url.path.endswith("/generation"):
            return httpx.Response(200, json={"data": generation or {}})
        body = json.loads(request.content)
        if sent is not None:
            sent.append(body)
        if handler is not None:
            return handler(request)
        if body.get("stream"):
            return (stream or _stream_response)(body)
        index = calls["messages"]
        calls["messages"] += 1
        return reply(index)

    return serve


def _install(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _probe(cli: Cli, bench_paths: BenchPaths, *args: str) -> Any:
    return cli.run("probe", *args, env={**bench_paths.env, **_ENV})


def _probe_trace(bench_paths: BenchPaths, prompts: tuple[int, ...] = (100, 200, 300)) -> Path:
    trace = bench_paths.traces_dir / "t.jsonl"
    _write_trace(trace, prompts)
    return trace


def test_probe_runs_a_rung_and_persists_the_run(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(
        monkeypatch, _transport(lambda index: _ok(cached=90, input_tokens=10) if index else _ok())
    )

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "2",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert set(doc) == {"run_dir", "run_hex", "conversation", "rungs", "summaries", "changed"}
    assert doc["conversation"] == "c1"
    assert doc["rungs"] == [1]
    assert len(str(doc["run_hex"])) == 12

    run_dir = Path(str(doc["run_dir"]))
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["protocol"] == "probe"
    assert meta["run_hex"] == doc["run_hex"]
    assert meta["options"]["rungs"] == [1]
    assert meta["options"]["repeats"] == [2, 2] or meta["options"]["repeats"] == [2]
    assert meta["prices"]["fake:model-a"]["source"] == "table"

    lines = (run_dir / "fake-model-a.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    # cold, two warm reads, then one streamed throughput request
    assert [record["role"] for record in records] == ["cold", "warm", "warm", "stream"]
    assert records[0]["cached"] == 0  # the cold write read nothing
    stream = records[-1]
    assert stream["usage"]["output_tokens"] == 18
    # a mock transport delivers the whole body at once, so there is a TTFT but no window
    # to compute a generation rate over; the real derivation is tested in test_bench_sse
    assert stream["ttft_ms"] is not None and stream["ttft_ms"] >= 0
    assert stream["fingerprint"] == " ".join(_STREAM_TOKENS[:16])  # thinking first, then text

    [summary] = [d for d in map(as_document, as_list(doc["summaries"]) or []) if d]
    assert summary["label"] == "fake:model-a"
    assert summary["hit_rate"] == 1.0
    assert summary["prefix_fraction"] == pytest.approx(0.9)
    assert summary["price_source"] == "table"
    assert summary["eff_per_m_prompt"] == pytest.approx((1 - 0.9) * 1.0 + 0.9 * 0.1)
    assert summary["errors"] == 0
    assert summary["skipped"] == 0


def test_a_compressed_trace_files_its_run_under_the_name_without_the_suffixes(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`t.jsonl.gz` is the trace `t`, and the packaged `sample.jsonl.gz` is the trace `sample`.

    The run directory is what `report` resolves a trace name to, so a name carrying the
    file's suffixes would be a run only a full path could reach.
    """
    _write_targets(bench_paths.targets_path)
    plain = _probe_trace(bench_paths)
    write_trace_text(bench_paths.traces_dir / "t.jsonl.gz", plain.read_text(encoding="utf-8"))
    _install(monkeypatch, _transport(lambda index: _ok()))

    outcome = _probe(cli, bench_paths, "t.jsonl.gz", "fake:model-a", "--gap", "0", "--yes")
    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    assert run_dir.parent == bench_paths.runs_dir / "t"

    # the name resolves back to that run, which is what a suffix in it would have broken
    reported = cli.run("report", "t", env={**bench_paths.env, **_ENV})
    assert reported.code == 0, reported.stderr
    assert reported.document["run_dir"] == str(run_dir)


def test_probe_defaults_to_the_smallest_middle_and_largest_rung(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths, prompts=(300, 100, 200, 400))
    _install(monkeypatch, _transport(lambda index: _ok()))

    outcome = _probe(cli, bench_paths, "t", "fake:model-a", "--gap", "0", "--yes")
    assert outcome.code == 0, outcome.stderr
    # candidates are turns 1-3; by recorded prompt: 2 (100), 3 (200), 1 (300)
    assert outcome.document["rungs"] == [1, 2, 3]


def test_probe_progress_lines_go_to_stderr_in_human_mode(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(monkeypatch, _transport(lambda index: _ok()))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    assert "fake:model-a rung 1 cold 0 200 0/100" in outcome.stderr
    assert "fake:model-a rung 1 warm 1 200" in outcome.stderr


def test_probe_human_output_has_the_two_tables(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(
        monkeypatch, _transport(lambda index: _ok(cached=90, input_tokens=10) if index else _ok())
    )

    outcome = cli.run(
        "probe",
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "2",
        "--gap",
        "0",
        "--yes",
        tty_stdout=True,
        env={**bench_paths.env, **_ENV},
    )
    assert outcome.code == 0, outcome.stderr
    assert "hit %" in outcome.stdout and "prefix %" in outcome.stdout
    assert "cached cold" in outcome.stdout and "hits" in outcome.stdout
    assert "90.0" in outcome.stdout  # the prefix fraction, as a percentage
    assert "0.9 0.9" in outcome.stdout  # the hit sequence, one cell per warm attempt
    assert all(len(line) <= 120 for line in outcome.stdout.splitlines())


def test_probe_estimate_is_shown_and_the_budget_refuses_above_it(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths, prompts=(100, 200, 300))
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    # rung 1: 100 cold + 2 * 200 warm + one streamed turn 2 (200 prompt, 256 output at
    # the table's output price, 2.0 USD/M) = 700 prompt tokens and 256 output tokens
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "2",
        "--gap",
        "0",
        "--budget",
        "0.0001",
        "--yes",
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "0.0012" in str(outcome.error["message"])
    assert sent == []
    assert not (bench_paths.runs_dir / "t").exists()

    within = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "2",
        "--gap",
        "0",
        "--budget",
        "0.002",
        "--yes",
    )
    assert within.code == 0, within.stderr
    assert "worst case" in within.stderr
    assert "700" in within.stderr


def test_probe_budget_refuses_a_spec_it_cannot_price(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    # targets.toml prices model-a only, so this spec's worst case cannot be computed
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-b",
        "--rungs",
        "1",
        "--gap",
        "0",
        "--budget",
        "1",
        "--yes",
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "no listed price for fake:model-b" in str(outcome.error["message"])
    assert "targets.toml" in str(outcome.error["hint"])
    assert sent == []


def test_probe_confirmation_names_a_spec_it_cannot_price(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(monkeypatch, _transport(lambda index: _ok()))

    outcome = cli.run(
        "probe",
        "t",
        "fake:model-b",
        "--rungs",
        "1",
        "--gap",
        "0",
        stdin="n\n",
        tty_stdin=True,
        tty_stderr=True,
        tty_stdout=False,
        env={**bench_paths.env, **_ENV},
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "confirmation_required"
    assert "no listed price for fake:model-b" in outcome.stderr


def test_probe_requires_confirmation_before_sending(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    outcome = _probe(cli, bench_paths, "t", "fake:model-a", "--rungs", "1", "--gap", "0")
    assert outcome.code == 2
    assert outcome.error["kind"] == "confirmation_required"
    assert sent == []
    assert not (bench_paths.runs_dir / "t").exists()


def test_probe_declined_prompt_sends_nothing(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    outcome = cli.run(
        "probe",
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--gap",
        "0",
        stdin="n\n",
        tty_stdin=True,
        tty_stderr=True,
        tty_stdout=False,
        env={**bench_paths.env, **_ENV},
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "confirmation_required"
    assert sent == []


def test_probe_rejects_an_impossible_rung_before_any_request(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    outcome = _probe(cli, bench_paths, "t", "fake:model-a", "--rungs", "9", "--gap", "0", "--yes")
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "rung 9 needs turn 10" in str(outcome.error["message"])
    assert sent == []


def test_probe_exits_one_when_a_request_failed_and_keeps_the_run(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)

    def reply(index: int) -> httpx.Response:
        if index == 0:
            return _ok()
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, _transport(reply))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",
        "--yes",
    )
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    assert "1 request(s) failed" in str(outcome.error["message"])
    assert "report" in str(outcome.error["hint"])

    run_dirs = sorted((bench_paths.runs_dir / "t").iterdir())
    assert len(run_dirs) == 1  # the run is on disk even though the command failed
    assert (run_dirs[0] / "run.json").is_file()
    records = (run_dirs[0] / "fake-model-a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    assert "boom" in records[1]  # the failed warm read is persisted, error and all


def test_probe_partial_failure_still_prints_the_tables_in_human_mode(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)

    def reply(index: int) -> httpx.Response:
        if index == 0:
            return _ok()
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, _transport(reply))
    outcome = cli.run(
        "probe",
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
        tty_stdout=True,
        env={**bench_paths.env, **_ENV},
    )
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    # the run cost money, so its numbers reach the terminal even though the call failed
    assert "hit %" in outcome.stderr and "cached cold" in outcome.stderr
    assert "fake:model-a" in outcome.stderr


def test_probe_partial_failure_gives_json_callers_the_summaries(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)

    def reply(index: int) -> httpx.Response:
        if index == 0:
            return _ok()
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, _transport(reply))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    context = as_document(outcome.error.get("context"))
    assert context is not None, "the error must carry the run's numbers"
    assert Path(str(context["run_dir"])).is_dir()
    assert context["conversation"] == "c1"
    [summary] = [d for d in map(as_document, as_list(context["summaries"]) or []) if d]
    assert summary["label"] == "fake:model-a"
    assert summary["errors"] == 1
    assert summary["rungs"] != []


def test_probe_exits_one_when_a_rung_is_skipped_by_a_failed_cold_write(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(monkeypatch, _transport(lambda index: httpx.Response(502, text="bad gateway")))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "2",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 1
    assert "1 rung(s) skipped" in str(outcome.error["message"])


def test_probe_warm_sends_no_nonce(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--warm",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    # the cold write, the warm read, and the throughput request all send the plain system
    assert [body["system"][0]["text"] for body in sent] == ["You are helpful."] * 3


def test_probe_prices_an_openrouter_spec_from_the_endpoint_snapshot(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(monkeypatch, _transport(lambda index: _ok()))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "or:model-a@novita",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    [summary] = [d for d in map(as_document, as_list(outcome.document["summaries"]) or []) if d]
    assert summary["price_source"] == "openrouter-endpoint"
    assert summary["input_price"] == pytest.approx(0.3)  # 0.0000003 USD/token, as USD per M

    meta = json.loads((Path(str(outcome.document["run_dir"])) / "run.json").read_text())
    snapshot = meta["endpoints"]["or:model-a@novita"]
    assert snapshot["tag"] == "novita"
    assert snapshot["quantization"] == "fp8"
    assert snapshot["context_length"] == 128000
    assert snapshot["price_input"] == pytest.approx(0.3)
    assert snapshot["price_cache_read"] == pytest.approx(0.03)
    assert snapshot["source"] == "openrouter-endpoint"


def test_report_summarises_a_probe_run_without_the_network(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(
        monkeypatch, _transport(lambda index: _ok(cached=90, input_tokens=10) if index else _ok())
    )
    probed = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    assert probed.code == 0, probed.stderr
    run_dir = Path(str(probed.document["run_dir"]))

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"report must not touch the network: {request.url}")

    _install(monkeypatch, explode)

    outcome = cli.run("report", str(run_dir), env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert doc["protocol"] == "probe"
    assert doc["summaries"] == []
    assert as_list(doc["probe_summaries"])
    [summary] = [d for d in map(as_document, as_list(doc["probe_summaries"]) or []) if d]
    assert summary["hit_rate"] == 1.0
    [rung] = [d for d in map(as_document, as_list(summary["rungs"]) or []) if d]
    assert rung["hits"] == [pytest.approx(0.9)]  # the warm read covered 90 % of the cold prompt

    rendered = cli.run("report", str(run_dir), tty_stdout=True, env=bench_paths.env)
    assert rendered.code == 0, rendered.stderr
    assert "hit %" in rendered.stdout and "cached cold" in rendered.stdout

    md_path = cli.root / "probe.md"
    marked = cli.run("report", str(run_dir), "--md", str(md_path), env=bench_paths.env)
    assert marked.code == 0, marked.stderr
    assert marked.document["changed"] is True
    assert md_path.read_text(encoding="utf-8").startswith("| spec |")


# --- the throughput request and TTL -------------------------------------------


def test_probe_no_throughput_sends_no_stream_request(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    assert [body["stream"] for body in sent] == [False, False]
    run_dir = Path(str(outcome.document["run_dir"]))
    records = (run_dir / "fake-model-a.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["role"] for line in records] == ["cold", "warm"]


def test_probe_streams_the_warm_body_with_a_real_output_budget(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(cached=90, input_tokens=10), sent=sent))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "2",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    cold, *warm = sent
    first_warm, *_ = warm
    stream = warm[-1]
    assert len(warm) == 3  # two warm reads, then the throughput request
    assert stream["stream"] is True
    assert stream["max_tokens"] == 256
    assert stream["temperature"] == 0
    # the prefix the streamed request reads is byte-identical to the warm reads'
    changed = {"stream", "max_tokens", "temperature"}
    assert {key: value for key, value in stream.items() if key not in changed} == {
        key: value for key, value in first_warm.items() if key not in changed
    }
    assert cold["stream"] is False

    run_dir = Path(str(outcome.document["run_dir"]))
    records = [
        json.loads(line) for line in (run_dir / "fake-model-a.jsonl").read_text().splitlines()
    ]
    assert [record["role"] for record in records] == ["cold", "warm", "warm", "stream"]
    [stream_record] = [record for record in records if record["role"] == "stream"]
    assert stream_record["fingerprint"] is not None
    assert stream_record["attempt"] == 0


def test_probe_streams_a_thinking_body_without_the_param_when_it_was_dropped(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped `thinking` key is dropped from the streamed body too."""
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    for seq, prompt_tokens in enumerate((100, 200, 300), start=1):
        entry = _trace_entry(seq, prompt_tokens)
        entry.body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        append_entry(trace, entry)
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            seen.append("thinking" in body)
            return _stream_response()
        if "thinking" in body:
            return httpx.Response(
                400, json={"error": {"message": "max_tokens must exceed thinking"}}
            )
        return _ok()

    _install(monkeypatch, _transport(lambda index: _ok(), handler=handler))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    assert seen == [False]


def test_probe_ttl_re_reads_the_first_rung_at_each_offset(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _install(monkeypatch, _transport(lambda index: _ok(cached=90, input_tokens=10)))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1,2",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--ttl",
        "5,10",
        "--no-throughput",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    records = [
        json.loads(line) for line in (run_dir / "fake-model-a.jsonl").read_text().splitlines()
    ]
    ttl = [record for record in records if record["role"] == "ttl"]
    assert [(record["rung"], record["attempt"]) for record in ttl] == [(1, 5), (1, 10)]
    assert [record["cached"] for record in ttl] == [90, 90]
    # rung 2 carries no TTL read, and the second offset waits past the first
    assert [record["rung"] for record in records if record["role"] == "cold"] == [1, 2]
    assert len(sleeps) == 2
    assert 0 < sleeps[0] <= 5 and sleeps[1] > sleeps[0]

    [summary] = [d for d in map(as_document, as_list(outcome.document["summaries"]) or []) if d]
    [first_rung, second_rung] = [d for d in map(as_document, as_list(summary["rungs"]) or []) if d]
    reads = [d for d in map(as_document, as_list(first_rung["ttl"]) or []) if d]
    assert [read["offset"] for read in reads] == [5, 10]
    assert all(read["hit"] is True for read in reads)
    assert as_list(second_rung["ttl"]) == []


def test_probe_rejects_a_gap_larger_than_a_ttl_offset(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--gap",
        "30",
        "--ttl",
        "10",
        "--yes",
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "--gap 30" in str(outcome.error["message"])
    assert sent == []


def test_probe_reports_a_failed_stream_without_skipping_the_rung(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(
        monkeypatch,
        _transport(
            lambda index: _ok(cached=90, input_tokens=10),
            stream=_refuse_stream,
        ),
    )
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    # the streamed request failed, so the command exits non-zero, but the rung is measured
    assert outcome.code == 1
    assert "1 request(s) failed" in str(outcome.error["message"])
    context = as_document(outcome.error.get("context"))
    assert context is not None
    [summary] = [d for d in map(as_document, as_list(context["summaries"]) or []) if d]
    assert summary["skipped"] == 0
    assert summary["hit_rate"] == 1.0
    assert summary["ttft_ms"] is None
    [rung] = [d for d in map(as_document, as_list(summary["rungs"]) or []) if d]
    assert as_list(rung["hits"]) == [pytest.approx(0.9)]


def test_probe_records_the_ttl_offsets_in_the_run(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    _install(monkeypatch, _transport(lambda index: _ok()))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--ttl",
        "5",
        "--no-throughput",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    run_json = Path(str(outcome.document["run_dir"])) / "run.json"
    meta = as_document(json.loads(run_json.read_text()))
    assert meta is not None
    options = as_document(meta["options"])
    assert options is not None
    assert options["ttl_s"] == [5]
    assert meta["ttl_s"] == [5]


def test_probe_reports_token_drift_and_a_model_mismatch_between_specs(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second spec answers with another model and a larger prompt, so drift names both.

    The prompt sizes come from the responses here, because the rung's size is what the
    provider says it read: the first spec is served turn 2 (200 tokens), the second spec the
    same turn from an endpoint that reports 400.
    """
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # the first spec sends three requests; the second spec's start at the fourth
        later = calls["n"] > 3
        model = "model-b" if later else "model-a"
        body = json.loads(request.content)
        if body.get("stream"):
            return _stream_response(model=model, input_tokens=310)
        if body["messages"][0]["content"] == "question 1":  # the rung's cold write
            # the rung's size is what the cold response reports: 200 for the second spec
            return _ok(input_tokens=200 if later else 100, model=model)
        return _ok(cached=90, input_tokens=310, model=model)

    _install(monkeypatch, _transport(lambda index: _ok(), handler=handler))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "fake:model-b",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    summaries = [d for d in map(as_document, as_list(outcome.document["summaries"]) or []) if d]
    first, second = summaries
    assert first["reference"] is True
    assert first["drift"] is None
    assert second["model_seen"] == "model-b"
    assert second["tokens_delta_pct"] == pytest.approx(100.0)  # rung 1 is 200 against 100
    assert second["drift"] == "model,tokens+100%"
    assert first["fingerprint_match"] is None  # the reference is never compared with itself


class _Clock:
    """A `perf_counter` stand-in the transport can move: the stream 'takes' 10 s."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_probe_ttl_counts_from_the_last_warm_read_not_from_the_stream(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 10 s streamed request is elapsed time, not added time: 30 s means 20 s of waiting."""
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    clock = _Clock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            clock.now += 10.0  # the generation the TTL offset has to account for
            return _stream_response()
        return _ok(cached=90, input_tokens=10)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("provibench.bench.probe.perf_counter", clock)
    _install(monkeypatch, _transport(lambda index: _ok(), handler=handler))

    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--ttl",
        "30,60",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    # the warm read finished at 100, the stream ended at 110, so 30 s and 60 s are 20 s and 50 s
    assert sleeps == [pytest.approx(20.0), pytest.approx(50.0)]


def test_probe_ttl_rides_the_first_served_rung(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rung 1's cold write failed, so the cache rung 2 wrote is the one to re-read."""
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths, prompts=(100, 200, 300, 400))
    seen = {"n": 0}

    def reply(index: int) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(502, text="bad gateway")  # rung 1's cold write fails
        return _ok(cached=90, input_tokens=10)

    _install(monkeypatch, _transport(reply))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1,2",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--ttl",
        "5",
        "--no-throughput",
        "--yes",
    )
    assert outcome.code == 1  # the failed cold request is reported, the run is kept
    context = as_document(outcome.error.get("context"))
    assert context is not None
    run_dir = Path(str(context["run_dir"]))
    records = [
        json.loads(line) for line in (run_dir / "fake-model-a.jsonl").read_text().splitlines()
    ]
    ttl = [record for record in records if record["role"] == "ttl"]
    assert [record["rung"] for record in ttl] == [2]
    assert [record["cached"] for record in ttl] == [90]


def test_probe_rejects_ttl_offsets_out_of_order(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(lambda index: _ok(), sent=sent))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--gap",
        "0",
        "--ttl",
        "300,60",
        "--yes",
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "ascend" in str(outcome.error["message"])
    assert sent == []


class _Stalled(httpx.AsyncByteStream):
    """A stream that sends one event and then goes quiet without closing."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"type": "message_start", "message": {"id": "m"}}\n\n'
        await asyncio.Event().wait()


def test_probe_does_not_hang_on_a_stream_that_stalls(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent stream is a failed request after `--timeout`, not a run that never ends."""
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200,
                stream=_Stalled(),
                headers={"content-type": "text/event-stream"},
            )
        return _ok(cached=90, input_tokens=10)

    _install(monkeypatch, _transport(lambda index: _ok(), handler=handler))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--timeout",
        "0.2",
        "--yes",
    )
    assert outcome.code == 1  # the stalled stream is reported, the rest of the run stands
    assert "1 request(s) failed" in str(outcome.error["message"])
    context = as_document(outcome.error.get("context"))
    assert context is not None
    [summary] = [d for d in map(as_document, as_list(context["summaries"]) or []) if d]
    assert summary["hit_rate"] == 1.0  # the warm read was served and is still scored
    assert summary["ttft_ms"] is None
    run_dir = Path(str(context["run_dir"]))
    records = [
        json.loads(line) for line in (run_dir / "fake-model-a.jsonl").read_text().splitlines()
    ]
    [stream] = [record for record in records if record["role"] == "stream"]
    assert "stream stalled" in str(stream["error"])


def test_probe_records_the_ttl_offset_it_actually_landed_on(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record carries both numbers: the offset asked for and the one it went out at."""
    _write_targets(bench_paths.targets_path)
    _probe_trace(bench_paths)
    clock = _Clock()

    async def fake_sleep(seconds: float) -> None:
        del seconds

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            clock.now += 10.0
            return _stream_response()
        return _ok(cached=90, input_tokens=10)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("provibench.bench.probe.perf_counter", clock)
    _install(monkeypatch, _transport(lambda index: _ok(), handler=handler))
    outcome = _probe(
        cli,
        bench_paths,
        "t",
        "fake:model-a",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--ttl",
        "30,60",
        "--yes",
    )
    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    records = [
        json.loads(line) for line in (run_dir / "fake-model-a.jsonl").read_text().splitlines()
    ]
    # the baseline is the warm read at 100 and the clock never moved again after the stream
    ttl = [record for record in records if record["role"] == "ttl"]
    assert [record["attempt"] for record in ttl] == [30, 60]
    assert [record["offset_actual_s"] for record in ttl] == [pytest.approx(10.0)] * 2

    [summary] = [d for d in map(as_document, as_list(outcome.document["summaries"]) or []) if d]
    [rung] = [d for d in map(as_document, as_list(summary["rungs"]) or []) if d]
    reads = [d for d in map(as_document, as_list(rung["ttl"]) or []) if d]
    assert [read["offset_actual_s"] for read in reads] == [pytest.approx(10.0)] * 2
