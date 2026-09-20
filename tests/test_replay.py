"""`replay`: spec/target/conversation wiring, the confirmation gate, and run persistence.

`bench.replay`'s own engine (retries, cost enrichment, cache math) is unit-tested in
`test_bench_replay.py` against a mocked transport; these tests exercise the CLI layer on
top of it, so the only boundary mocked here is the network `httpx.AsyncClient` opens
(`_client_factory` below), never the replay logic itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from provibench.bench.trace import (
    RecordedResponse,
    TraceEntry,
    Usage,
    append_entry,
    read_trace_text,
    write_trace_text,
)
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


def _trace_entry(
    seq: int, conversation: str = "c1", text: str = "hi", *, prompt_tokens: int = 100
) -> TraceEntry:
    """One recorded turn: a system prompt (so the run nonce has somewhere to go) and a usage."""
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
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1024,
        },
        conversation=conversation,
        response=RecordedResponse(
            status=200, latency_ms=1.0, usage=Usage(input_tokens=prompt_tokens)
        ),
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


def _capture(bodies: list[dict[str, Any]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok_response(request)

    return handler


def _run_dir(outcome: Any) -> Path:
    return Path(str(outcome.document["run_dir"]))


def _meta(run_dir: Path) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return raw


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
    assert doc["partial"] is False
    assert doc["changed"] is True
    run_dir = _run_dir(outcome)
    assert (run_dir / "run.json").is_file()
    [summary] = [d for d in map(as_document, as_list(doc["summaries"]) or []) if d]
    assert summary["label"] == "t:model-a"
    assert summary["ok"] == 2
    assert summary["errors"] == 0
    cost = as_document(summary["cost"]) or {}
    assert cost["source"] == "targets"


def test_replay_exits_non_zero_on_a_failed_turn(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed turn is F1c/O5a: `partial: true` still reaches stdout, exit is non-zero, and
    the `operation_failed` error's context names only the run, not the document again."""
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1, prompt_tokens=100))
    append_entry(trace, _trace_entry(2, prompt_tokens=100))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return _ok_response(request)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    context = as_document(outcome.error.get("context"))
    assert context is not None
    assert set(context) == {"run_dir", "run_hex"}
    doc = outcome.document
    assert doc["partial"] is True
    [summary] = [d for d in map(as_document, as_list(doc["summaries"]) or []) if d]
    assert summary["errors"] == 1


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


def test_replay_files_a_gzipped_trace_under_its_plain_name(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    plain = bench_paths.traces_dir / "t.jsonl"
    append_entry(plain, _trace_entry(1))
    write_trace_text(bench_paths.traces_dir / "t.jsonl.gz", read_trace_text(plain))
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok_response))

    outcome = cli.run(
        "replay",
        "t.jsonl.gz",
        "--run",
        "t:model-a",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )

    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    assert run_dir.parent == bench_paths.runs_dir / "t"


def test_replay_of_the_packaged_sample_files_under_sample(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok_response))

    outcome = cli.run(
        "replay",
        "sample-trace",
        "--run",
        "t:model-a",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )

    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    assert run_dir.parent == bench_paths.runs_dir / "sample-trace"
    assert outcome.document["turns"] == 30


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


# --- the run nonce ------------------------------------------------------------


def test_replay_stamps_one_run_nonce_into_every_turn_and_records_it(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))
    append_entry(trace, _trace_entry(2))
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_capture(bodies)))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 0, outcome.stderr

    nonces = [body["system"][0]["text"].splitlines()[0] for body in bodies]
    assert len(nonces) == 2
    assert len(set(nonces)) == 1  # one nonce per run, so turn 2 can read turn 1's write
    assert nonces[0].startswith("provibench-run:")
    meta = _meta(_run_dir(outcome))
    assert meta["protocol"] == "full"
    assert meta["run_hex"] == nonces[0].removeprefix("provibench-run:")
    assert meta["options"]["warm"] is False
    assert bodies[0]["system"][0]["text"].endswith("You are helpful.")
    # the nonce goes into the first block only; the recorded marker block is untouched
    assert bodies[0]["system"][1] == {
        "type": "text",
        "text": "Repo context.",
        "cache_control": {"type": "ephemeral"},
    }


def test_replay_warm_sends_no_nonce_and_records_the_choice(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1))
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_capture(bodies)))

    outcome = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        "--warm",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert outcome.code == 0, outcome.stderr
    assert bodies[0]["system"][0]["text"] == "You are helpful."
    assert _meta(_run_dir(outcome))["options"]["warm"] is True


def test_replay_without_a_system_prompt_fails_before_any_request(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    entry = _trace_entry(1)
    entry.body.pop("system")
    append_entry(trace, entry)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok_response(request)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    assert "--warm" in str(outcome.error["message"])
    assert calls["n"] == 0


# --- estimate and budget ------------------------------------------------------


def test_replay_shows_the_worst_case_estimate_before_sending(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1, prompt_tokens=200))
    append_entry(trace, _trace_entry(2, prompt_tokens=300))
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok_response))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", "--yes", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 0, outcome.stderr
    assert "worst case" in outcome.stderr
    assert "500" in outcome.stderr  # 200 + 300 recorded prompt tokens
    assert "0.0005" in outcome.stderr  # 500 tokens at 1.0 USD/M


def test_replay_budget_refuses_above_the_estimate_and_sends_nothing(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1, prompt_tokens=200))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok_response(request)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    outcome = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        "--budget",
        "0.0001",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert calls["n"] == 0
    assert not (bench_paths.runs_dir / "t").exists()


def test_replay_dry_run_sends_nothing_and_reports_the_estimate(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--dry-run` prices the run and stops: no request, `requires_confirmation` (R4)."""
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1, prompt_tokens=200))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok_response(request)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))

    outcome = cli.run(
        "replay", "t", "--run", "t:model-a", "--dry-run", env={**bench_paths.env, "X_KEY": "secret"}
    )
    assert outcome.code == 0, outcome.stderr
    assert calls["n"] == 0
    assert not (bench_paths.runs_dir / "t").exists()
    doc = outcome.document
    assert doc["partial"] is False
    assert doc["changed"] is False
    assert doc["requires_confirmation"] is True
    assert doc["runs_dir"] == str(bench_paths.runs_dir)
    assert "run_dir" not in doc and "summaries" not in doc
    [estimate] = [d for d in map(as_document, as_list(doc["estimate"]) or []) if d]
    assert estimate["label"] == "t:model-a"
    assert estimate["worst_case_usd"] == pytest.approx(0.0002)

    with_yes = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        "--dry-run",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert with_yes.code == 0, with_yes.stderr
    assert calls["n"] == 0
    assert with_yes.document["requires_confirmation"] is True


def test_replay_budget_within_the_estimate_still_runs(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _trace_entry(1, prompt_tokens=200))
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok_response))

    outcome = cli.run(
        "replay",
        "t",
        "--run",
        "t:model-a",
        "--budget",
        "1.0",
        "--yes",
        env={**bench_paths.env, "X_KEY": "secret"},
    )
    assert outcome.code == 0, outcome.stderr
