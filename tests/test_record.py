"""`record`: proxy lifecycle wiring, existing-trace refusal, and --append.

The recording proxy itself (`bench.proxy.Recorder`) opens real sockets and is out of scope
for a CLI test; `provibench.bench.proxy.serve` is the network/process boundary the command
calls, and is faked here to append trace entries the way the proxy would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from provibench.bench.trace import TraceEntry, append_entry
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli


def _entry(seq: int, conversation: str = "c1", text: str = "hi") -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        body={"messages": [{"role": "user", "content": text}]},
        conversation=conversation,
    )


def test_record_starts_the_proxy_and_summarises_by_conversation(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_serve(upstream: str, trace_path: Path, host: str, port: int, **kwargs: object) -> None:
        append_entry(trace_path, _entry(1, "c1"))
        append_entry(trace_path, _entry(2, "c2"))

    monkeypatch.setattr("provibench.bench.proxy.serve", fake_serve)
    outcome = cli.run(
        "record", "--name", "sess", "--upstream", "https://up.test", env=bench_paths.env
    )
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert doc["trace"] == str(bench_paths.traces_dir / "sess.jsonl")
    assert doc["requests"] == 2
    assert doc["changed"] is True
    entries = [d for d in map(as_document, as_list(doc["conversations"]) or []) if d]
    conversations = {c["key"]: c for c in entries}
    assert conversations["c1"]["requests"] == 1
    assert conversations["c2"]["requests"] == 1


def test_record_refuses_an_existing_trace_without_append(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = bench_paths.traces_dir / "sess.jsonl"
    append_entry(trace_path, _entry(1))
    called = False

    def fake_serve(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("provibench.bench.proxy.serve", fake_serve)
    outcome = cli.run(
        "record", "--name", "sess", "--upstream", "https://up.test", env=bench_paths.env
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert not called
    assert len(trace_path.read_text().splitlines()) == 1  # untouched


def test_record_append_adds_to_an_existing_trace(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = bench_paths.traces_dir / "sess.jsonl"
    append_entry(trace_path, _entry(1))

    def fake_serve(upstream: str, trace_path: Path, host: str, port: int, **kwargs: object) -> None:
        append_entry(trace_path, _entry(2))

    monkeypatch.setattr("provibench.bench.proxy.serve", fake_serve)
    outcome = cli.run(
        "record",
        "--name",
        "sess",
        "--upstream",
        "https://up.test",
        "--append",
        env=bench_paths.env,
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["requests"] == 2
    assert outcome.document["changed"] is True


def test_record_swallows_keyboard_interrupt_and_reports_what_was_captured(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_serve(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("provibench.bench.proxy.serve", fake_serve)
    outcome = cli.run(
        "record", "--name", "sess", "--upstream", "https://up.test", env=bench_paths.env
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.document == {
        "trace": str(bench_paths.traces_dir / "sess.jsonl"),
        "requests": 0,
        "conversations": [],
        "changed": False,
    }
    assert bench_paths.traces_dir.is_dir()


def test_record_progress_messages_go_to_stderr(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_serve(upstream: str, trace_path: Path, host: str, port: int, **kwargs: object) -> None:
        append_entry(trace_path, _entry(1))

    monkeypatch.setattr("provibench.bench.proxy.serve", fake_serve)
    outcome = cli.run(
        "record",
        "--name",
        "sess",
        "--upstream",
        "https://up.test",
        "--host",
        "0.0.0.0",
        "--port",
        "9999",
        env=bench_paths.env,
    )
    assert outcome.code == 0
    assert "listen http://0.0.0.0:9999" in outcome.stderr
