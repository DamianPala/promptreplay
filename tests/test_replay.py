"""`replay`: spec/target/conversation wiring, the confirmation gate, and run persistence.

`bench.replay`'s own engine (retries, cost enrichment, cache math) is unit-tested in
`test_bench_replay.py` against a mocked transport; these tests exercise the CLI layer on
top of it, so the only boundary mocked here is the network `httpx.AsyncClient` opens
(`_client_factory` below), never the replay logic itself.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from provibench.bench.trace import TraceEntry, append_entry
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli

_RealAsyncClient = httpx.AsyncClient

_TARGETS_TOML = (
    "[targets.t]\n"
    'url = "https://x.test/v1/messages"\n'
    'api_key_env = "X_KEY"\n'
    'kind = "anthropic"\n'
    "\n"
    '[targets.t.prices."model-a"]\n'
    "input = 1.0\n"
    "cache_read = 0.1\n"
    "cache_write = 1.0\n"
    "output = 2.0\n"
)


def _write_targets(path: Path) -> None:
    path.write_text(_TARGETS_TOML)


def _trace_entry(seq: int, conversation: str = "c1", text: str = "hi") -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        body={"model": "orig", "messages": [{"role": "user", "content": text}], "max_tokens": 1024},
        conversation=conversation,
    )


def _client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.AsyncClient]:
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)  # type: ignore[arg-type]

    return factory


def _ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "model": "model-a",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )


def test_replay_runs_and_persists_a_run(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))
    append_entry(trace, _trace_entry(2))
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok_response))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert doc["conversation"] == "c1"
    assert doc["turns"] == 2
    assert doc["changed"] is True
    run_dir = Path(str(doc["run_dir"]))
    assert (run_dir / "run.json").is_file()
    [summary] = [d for d in map(as_document, as_list(doc["summaries"]) or []) if d]
    assert summary["label"] == "t:model-a"
    assert summary["ok"] == 2
    assert summary["errors"] == 0
    cost = as_document(summary["cost"]) or {}
    assert cost["source"] == "table"


def test_replay_limit_truncates_the_replayed_turns(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))
    append_entry(trace, _trace_entry(2))
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok_response))

    outcome = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        "--limit",
        "1",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["turns"] == 1


def test_replay_selects_conversation_and_rejects_unknown_one(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1, conversation="c1"))
    append_entry(trace, _trace_entry(2, conversation="c2"))

    missing = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        "--conversation",
        "ghost",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert missing.code == 1
    assert missing.error["kind"] == "not_found"


def test_replay_unknown_target_in_run_spec_is_invalid_input(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))

    outcome = cli.run(
        "replay", "t", "--run", "ghost:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "ghost" in str(outcome.error["message"])


def test_replay_missing_api_key_is_operation_failed(cli: Cli, bench_paths: BenchPaths) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))

    outcome = cli.run("replay", "t", "--run", "t:model-a", "--yes", env=bench_paths.env)
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    assert "X_KEY" in str(outcome.error["message"])


def test_replay_without_yes_requires_confirmation_when_it_cannot_prompt(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "confirmation_required"
    assert not (bench_paths.runs_dir / "t").exists()


def test_replay_declined_prompt_is_confirmation_required(cli: Cli, bench_paths: BenchPaths) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))

    outcome = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        stdin="n\n",
        tty_stdin=True,
        tty_stderr=True,
        tty_stdout=False,
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "confirmation_required"


def test_replay_missing_trace_is_not_found(cli: Cli, bench_paths: BenchPaths) -> None:
    _write_targets(bench_paths.targets_path)
    outcome = cli.run(
        "replay", "nope", "--run", "t:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"
