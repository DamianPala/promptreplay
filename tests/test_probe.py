"""`probe` and `report`: the estimate, the budget, the run it persists, and the report.

The only boundary mocked is the network `httpx.AsyncClient` opens, so these tests exercise
the command, the probe protocol and the aggregation together, on recorded fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from provibench.bench.trace import RecordedResponse, TraceEntry, Usage, append_entry
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


def _ok(cached: int = 0, input_tokens: int = 100) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "model": "model-a",
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cached,
                "output_tokens": 1,
            },
        },
    )


def _transport(
    reply: Callable[[int], httpx.Response],
    *,
    sent: list[dict[str, Any]] | None = None,
    generation: dict[str, Any] | None = _GENERATION,
) -> Callable[[httpx.Request], httpx.Response]:
    """A transport serving /endpoints and /generation, and delegating the rest to `reply`."""
    calls = {"messages": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=_ENDPOINTS)
        if request.url.path.endswith("/generation"):
            return httpx.Response(200, json={"data": generation or {}})
        if sent is not None:
            sent.append(json.loads(request.content))
        index = calls["messages"]
        calls["messages"] += 1
        return reply(index)

    return handler


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
    assert [record["role"] for record in records] == ["cold", "warm", "warm"]
    assert records[0]["cached"] == 0  # the cold write read nothing

    [summary] = [d for d in map(as_document, as_list(doc["summaries"]) or []) if d]
    assert summary["label"] == "fake:model-a"
    assert summary["hit_rate"] == 1.0
    assert summary["prefix_fraction"] == pytest.approx(0.9)
    assert summary["price_source"] == "table"
    assert summary["eff_per_m_prompt"] == pytest.approx((1 - 0.9) * 1.0 + 0.9 * 0.1)
    assert summary["errors"] == 0
    assert summary["skipped"] == 0


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

    # rung 1, two warm reads: 100 cold + 2 * 200 warm = 500 tokens at 1.0 USD/M
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
    assert "0.0005" in str(outcome.error["message"])
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
        "0.001",
        "--yes",
    )
    assert within.code == 0, within.stderr
    assert "worst case" in within.stderr
    assert "500" in within.stderr


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
    assert [body["system"][0]["text"] for body in sent] == ["You are helpful."] * 2


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
    [endpoint] = meta["endpoints"]["model-a"]
    assert endpoint["tag"] == "novita"
    assert endpoint["quantization"] == "fp8"


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
