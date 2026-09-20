"""`inspect`: trace resolution (path or name under traces_dir), conversation selection."""

from __future__ import annotations

from provibench.bench.trace import RecordedResponse, TraceEntry, Usage, append_entry
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli


def _entry(
    seq: int, conversation: str, text: str, *, response: RecordedResponse | None = None
) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        body={"messages": [{"role": "user", "content": text}]},
        conversation=conversation,
        response=response,
    )


def test_inspect_selects_the_conversation_with_the_most_bytes_by_default(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    response = RecordedResponse(
        status=200, latency_ms=12.0, ttft_ms=1.0, provider="prov", usage=Usage(input_tokens=10)
    )
    append_entry(trace, _entry(1, "main", "x" * 500, response=response))
    append_entry(trace, _entry(2, "side", "y"))

    outcome = cli.run("inspect", "t", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert doc["trace"] == str(trace)
    assert doc["selected"] == "main"
    conversations = [d for d in map(as_document, as_list(doc["conversations"]) or []) if d]
    assert {c["key"] for c in conversations} == {"main", "side"}
    turns = [d for d in map(as_document, as_list(doc["turns"]) or []) if d]
    [turn] = turns
    assert turn["status"] == 200
    assert turn["provider"] == "prov"
    assert turn["input"] == 10


def test_inspect_conversation_flag_selects_and_rejects_unknown_key(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _entry(1, "main", "x" * 10))
    append_entry(trace, _entry(2, "side", "y"))

    selected = cli.run("inspect", "t", "--conversation", "side", env=bench_paths.env)
    assert selected.code == 0
    assert selected.document["selected"] == "side"
    assert len(as_list(selected.document["turns"]) or []) == 1

    missing = cli.run("inspect", "t", "--conversation", "ghost", env=bench_paths.env)
    assert missing.code == 1
    assert missing.error["kind"] == "not_found"


def test_inspect_accepts_an_explicit_trace_path(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir.parent / "custom.jsonl"
    append_entry(trace, _entry(1, "c1", "hi"))
    outcome = cli.run("inspect", str(trace), env=bench_paths.env)
    assert outcome.code == 0
    assert outcome.document["trace"] == str(trace)


def test_inspect_resolves_a_relative_path_against_the_injected_cwd(cli: Cli) -> None:
    # Regression test: `resolve_trace_path` must join relative paths against
    # `invocation.cwd` rather than the real process cwd, or this only works by accident
    # when the two happen to match (as they do outside the test harness).
    trace = cli.root / "custom.jsonl"
    append_entry(trace, _entry(1, "c1", "hi"))
    outcome = cli.run("inspect", "custom.jsonl")
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["trace"] == str(trace)


def test_inspect_missing_trace_is_not_found(cli: Cli, bench_paths: BenchPaths) -> None:
    outcome = cli.run("inspect", "nope", env=bench_paths.env)
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"


def test_inspect_empty_trace_has_no_selected_conversation(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "empty.jsonl"
    trace.touch()
    outcome = cli.run("inspect", "empty", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert outcome.document == {
        "trace": str(trace),
        "conversations": [],
        "selected": None,
        "turns": [],
    }


def test_inspect_resolves_the_packaged_sample(cli: Cli, bench_paths: BenchPaths) -> None:
    outcome = cli.run("inspect", "sample-trace", env=bench_paths.env)

    assert outcome.code == 0, outcome.stderr
    document = outcome.document
    assert str(document["trace"]).endswith("sample-trace.jsonl.gz")
    assert document["selected"] is not None
    assert len(as_list(document["turns"]) or []) == 30


def test_inspect_reads_a_gzipped_trace(cli: Cli, bench_paths: BenchPaths) -> None:
    from provibench.bench.trace import read_trace_text, write_trace_text

    plain = bench_paths.traces_dir / "t.jsonl"
    append_entry(plain, _entry(1, "main", "hi"))
    trace = bench_paths.traces_dir / "t.jsonl.gz"
    write_trace_text(trace, read_trace_text(plain))

    outcome = cli.run("inspect", "t.jsonl.gz", env=bench_paths.env)

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["trace"] == str(trace)
    assert len(as_list(outcome.document["turns"]) or []) == 1


def test_inspect_human_table(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _entry(1, "main", "hi"))
    outcome = cli.run("inspect", "t", tty=True, env=bench_paths.env)
    assert outcome.code == 0
    assert "conversations:" in outcome.stdout
    assert "turns: main" in outcome.stdout


def test_inspect_text_view_shows_prompt_tokens_and_drops_seqs(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """Item 3: the text view trades first_seq/last_seq for a `first -> last` token column;
    the JSON keeps every field."""
    trace = bench_paths.traces_dir / "t.jsonl"
    first = RecordedResponse(status=200, latency_ms=1.0, ttft_ms=1.0, usage=Usage(input_tokens=5))
    last = RecordedResponse(status=200, latency_ms=1.0, ttft_ms=1.0, usage=Usage(input_tokens=50))
    append_entry(trace, _entry(1, "main", "x" * 500, response=first))
    append_entry(trace, _entry(2, "main", "x" * 500, response=last))

    outcome = cli.run("inspect", "t", tty=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert "first_seq" not in outcome.stdout
    assert "last_seq" not in outcome.stdout
    assert "5 → 50" in outcome.stdout

    json_outcome = cli.run("inspect", "t", "--json", env=bench_paths.env)
    [conversation] = [
        d for d in map(as_document, as_list(json_outcome.document["conversations"]) or []) if d
    ]
    assert conversation["first_seq"] == 1
    assert conversation["last_seq"] == 2
    assert conversation["prompt_tokens_first"] == 5
    assert conversation["prompt_tokens_last"] == 50


def test_inspect_text_view_drops_provider_column_when_no_turn_has_one(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """Item 3: a native trace (no turn carries a provider) drops that column from the text
    view; the JSON keeps the field, null, on every turn."""
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _entry(1, "main", "hi"))

    outcome = cli.run("inspect", "t", tty=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert "provider" not in outcome.stdout

    json_outcome = cli.run("inspect", "t", "--json", env=bench_paths.env)
    [turn] = [d for d in map(as_document, as_list(json_outcome.document["turns"]) or []) if d]
    assert turn["provider"] is None


def test_inspect_blank_line_separates_the_two_sections(cli: Cli, bench_paths: BenchPaths) -> None:
    """Item 2's layout rule: one blank line between the conversations section and the
    turns section, no blank line inside either, and no leading or trailing blank."""
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _entry(1, "main", "hi"))

    outcome = cli.run("inspect", "t", tty=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    lines = outcome.stdout.splitlines()
    assert lines[0] != ""
    assert lines[-1] != ""
    blanks = [i for i, line in enumerate(lines) if line == ""]
    assert blanks == [lines.index("turns: main") - 1]
