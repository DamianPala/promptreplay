"""`scrub`: what the copy contains, what the report says, and how OUT is guarded."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from promptreplay.bench.trace import (
    RecordedResponse,
    TraceEntry,
    Usage,
    append_entry,
    load_trace,
    read_trace_text,
    write_trace_text,
)
from promptreplay.core.documents import Document, as_document, as_list
from tests.conftest import BenchPaths, Cli, Outcome

_ANTHROPIC_KEY = "sk-ant-api03-FAKE0000000000000000"
_EMAIL = "operator@example.com"
_HOME = "/home/operator"


def _entry(
    seq: int, conversation: str, content: str, *, response: RecordedResponse | None = None
) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        headers={"authorization": "Bearer abc-1234567890", "x-api-key": _ANTHROPIC_KEY},
        body={
            "metadata": {"user_id": "operator"},
            "system": [
                {
                    "type": "text",
                    "text": f"Instructions live in {_HOME}/AGENTS.md",
                    "metadata": {"user_id": "operator"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": f"wrote {_HOME}/out.txt",
                            "metadata": {"user_id": "operator"},
                        },
                        {"type": "text", "text": f"mail {_EMAIL}"},
                    ],
                }
            ],
            "marker": content,
        },
        conversation=conversation,
        response=response,
    )


def _fixture(trace: Path) -> None:
    """One main conversation, one side conversation, secrets in body, headers, response."""
    response = RecordedResponse(
        status=200,
        latency_ms=12.0,
        ttft_ms=1.0,
        provider="prov",
        usage=Usage(input_tokens=10, output_tokens=2, cache_read_input_tokens=3),
        error=f"failed reading {_HOME}/secret.txt",
    )
    append_entry(trace, _entry(1, "main", "x" * 500, response=response))
    append_entry(trace, _entry(2, "main", "y" * 500))
    append_entry(trace, _entry(3, "side", "z"))


def _lines(path: Path) -> list[Document]:
    return [
        as_document(json.loads(line)) or {} for line in read_trace_text(path).splitlines() if line
    ]


def _run(cli: Cli, bench_paths: BenchPaths, *args: str) -> Outcome:
    return cli.run("scrub", *args, env=bench_paths.env)


def test_scrub_writes_a_clean_copy_and_reports_what_it_removed(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    document = outcome.document
    assert document["trace"] == str(trace)
    assert document["out"] == str(out)
    assert document["entries_in"] == 3
    assert document["entries_out"] == 3
    assert document["entries_dropped"] == 0
    assert document["user_names"] == ["operator"]
    assert document["changed"] is True
    rules = {str(row["rule"]): row["count"] for row in _rules(document)}
    assert rules["path"] == 7  # six body paths plus the one inside the recorded response
    assert rules["metadata"] == 3
    assert rules["anthropic-key"] == 3
    assert rules["bearer"] == 3
    assert rules["email"] == 3
    assert rules["aws-key"] == 0

    written = read_trace_text(out)
    assert _HOME not in written
    assert _EMAIL not in written
    assert _ANTHROPIC_KEY not in written
    assert "Bearer [scrubbed:bearer]" in written
    assert "[scrubbed:anthropic-key]" in written
    assert "[scrubbed:email]" in written
    assert "/home/user/AGENTS.md" in written

    first = _lines(out)[0]
    assert first["seq"] == 1 and first["ts"] == "2026-01-01T00:00:01Z"
    body = as_document(first["body"]) or {}
    assert "metadata" not in body
    response = as_document(first["response"]) or {}
    assert response["error"] == "failed reading /home/user/secret.txt"
    usage = as_document(response["usage"]) or {}
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    }
    # The nested metadata block stays, with its strings scrubbed like any other.
    system = as_list(body["system"]) or []
    assert system == [
        {
            "type": "text",
            "text": "Instructions live in /home/user/AGENTS.md",
            "metadata": {"user_id": "operator"},
        }
    ]


def test_scrub_output_file_writes_the_result_document_and_keeps_stdout_empty(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)
    product = cli.root / "clean.jsonl"
    result_path = cli.root / "summary.json"

    outcome = _run(
        cli,
        bench_paths,
        "t",
        str(product),
        "--output-file",
        str(result_path),
    )

    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout == ""
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert document["out"] == str(product)
    assert document["output_file"] == str(result_path)


def _rules(document: Document) -> list[Document]:
    return [row for row in map(as_document, as_list(document["rules"]) or []) if row is not None]


def test_scrub_is_byte_identical_for_the_same_input(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)
    first = cli.root / "first.jsonl.gz"
    second = cli.root / "second.jsonl.gz"

    assert _run(cli, bench_paths, "t", str(first)).code == 0
    assert _run(cli, bench_paths, "t", str(second)).code == 0

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[:2] == b"\x1f\x8b"
    # A .gz output is a trace like any other.
    assert len(load_trace(first)) == 3


def test_scrub_reads_a_gzipped_trace(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl.gz"
    write_trace_text(
        trace,
        "".join(
            f"{entry.model_dump_json(exclude_none=True)}\n" for entry in [_entry(1, "main", "x")]
        ),
    )
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t.jsonl.gz", str(out))

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["entries_in"] == 1
    assert _HOME not in read_trace_text(out)


def test_scrub_turns_keeps_the_main_conversation_and_reports_the_drop(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(trace, _entry(1, "side", "z"))
    append_entry(trace, _entry(2, "main", "x" * 500))
    append_entry(trace, _entry(3, "main", "y" * 500))
    append_entry(trace, _entry(4, "main", "q" * 500))
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out), "--turns", "2")

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["entries_in"] == 4
    assert outcome.document["entries_out"] == 2
    assert outcome.document["entries_dropped"] == 2
    assert outcome.document["selection"] == "conversation"
    assert [line["seq"] for line in _lines(out)] == [2, 3]


def test_scrub_turns_without_conversations_keeps_the_first_entries_in_file_order(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    trace.write_text(
        "".join(json.dumps({"seq": seq, "body": {"text": "x" * 40}}) + "\n" for seq in (1, 2, 3)),
        encoding="utf-8",
    )
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out), "--turns", "2")

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["selection"] == "order"
    assert outcome.document["entries_in"] == 3
    assert outcome.document["entries_out"] == 2
    assert outcome.document["entries_dropped"] == 1
    assert [line["seq"] for line in _lines(out)] == [1, 2]


def test_scrub_user_replaces_whole_words(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(
        trace,
        TraceEntry(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            path="/v1/messages",
            body={
                "messages": [
                    {
                        "role": "user",
                        "content": "ls -l: -rw-rw-r-- 1 operator operator 42 out.txt",
                    }
                ]
            },
            conversation="main",
        ),
    )
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out), "--user", "operator")

    assert outcome.code == 0, outcome.stderr
    rules = {str(row["rule"]): row["count"] for row in _rules(outcome.document)}
    assert rules["user"] == 2
    written = read_trace_text(out)
    assert "-rw-rw-r-- 1 user user 42 out.txt" in written
    assert "operator" not in written


def test_scrub_rewrites_the_encoded_project_path(cli: Cli, bench_paths: BenchPaths) -> None:
    """The encoded form shares the entry's name even when it sits in another entry."""
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(
        trace,
        TraceEntry(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            path="/v1/messages",
            body={"messages": [{"role": "user", "content": "memory -home-operator-recwork/m"}]},
            conversation="main",
        ),
    )
    append_entry(trace, _entry(2, "main", "z"))
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    written = read_trace_text(out)
    assert "-home-user-recwork/m" in written
    assert "-home-operator" not in written
    assert "/home/operator" not in written


def test_scrub_merging_keys_in_one_entry_is_an_input_error(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    trace.write_text(
        '{"seq": 9, "conversation": "main", "body": {"/home/one/x": 1, "/home/two/x": 2}}\n',
        encoding="utf-8",
    )

    outcome = _run(cli, bench_paths, "t", str(cli.root / "clean.jsonl"))

    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "9" in str(outcome.error["message"])


def test_scrub_replace_applies_literally(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(
        trace,
        TraceEntry(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            path="/v1/messages",
            body={
                "messages": [{"role": "user", "content": f"acme-corp on build-07 in {_HOME}/work"}]
            },
            conversation="main",
        ),
    )
    out = cli.root / "clean.jsonl"

    outcome = _run(
        cli,
        bench_paths,
        "t",
        str(out),
        "--replace",
        "acme-corp=example",
        "--replace",
        "build-07=host",
    )

    assert outcome.code == 0, outcome.stderr
    rules = {str(row["rule"]): row["count"] for row in _rules(outcome.document)}
    assert rules["replace:acme-corp"] == 1
    assert rules["replace:build-07"] == 1
    written = read_trace_text(out)
    assert "example on host in /home/user/work" in written
    assert "acme-corp" not in written


def test_scrub_allow_email_keeps_the_address(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out), "--allow-email", _EMAIL)

    assert outcome.code == 0, outcome.stderr
    written = read_trace_text(out)
    assert written.count(_EMAIL) == 3
    assert "[scrubbed:email]" not in written


def test_scrub_refuses_to_overwrite_without_force(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)
    out = cli.root / "clean.jsonl"
    out.write_text("keep me\n", encoding="utf-8")

    refused = _run(cli, bench_paths, "t", str(out))

    assert refused.code == 1
    assert refused.error["kind"] == "precondition_failed"
    assert out.read_text(encoding="utf-8") == "keep me\n"

    forced = _run(cli, bench_paths, "t", str(out), "--force")
    assert forced.code == 0, forced.stderr
    assert _HOME not in read_trace_text(out)


def test_scrub_out_must_differ_from_the_trace(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)

    outcome = _run(cli, bench_paths, "t", str(trace), "--force")

    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"


def test_scrub_creates_the_output_directory(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)
    out = cli.root / "nested" / "deeper" / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    assert out.is_file()


def test_scrub_nothing_to_remove_still_exits_zero(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    append_entry(
        trace,
        TraceEntry(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            path="/v1/messages",
            body={"messages": [{"role": "user", "content": "nothing sensitive here"}]},
            conversation="main",
        ),
    )
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["changed"] is False
    assert all(row["count"] == 0 for row in _rules(outcome.document))
    assert read_trace_text(out) == read_trace_text(trace)


def test_scrub_reports_no_change_when_only_the_formatting_differs(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """An input written with spaces re-serialises smaller; that is not a removal."""
    trace = bench_paths.traces_dir / "t.jsonl"
    entry = {
        "seq": 1,
        "ts": "2026-01-01T00:00:00Z",
        "path": "/v1/messages",
        "body": {"messages": [{"role": "user", "content": "nothing sensitive here"}]},
        "conversation": "main",
    }
    trace.write_text(f"{json.dumps(entry)}\n", encoding="utf-8")
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["bytes_in"] != outcome.document["bytes_out"]
    assert outcome.document["changed"] is False


def test_scrub_human_table_lists_the_rules(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)

    outcome = cli.run(
        "scrub",
        "t",
        str(cli.root / "clean.jsonl"),
        tty=True,
        env=bench_paths.env,
    )

    assert outcome.code == 0, outcome.stderr
    assert "scrubbed:" in outcome.stdout
    for label in ("rule", "path", "anthropic-key", "email", "metadata", "user"):
        assert label in outcome.stdout
    assert "User names: operator" in outcome.stdout
    assert "Selection: every entry" in outcome.stdout


def test_scrub_human_output_has_no_leading_or_trailing_blank_line(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)

    outcome = cli.run("scrub", "t", str(cli.root / "clean.jsonl"), tty=True, env=bench_paths.env)

    assert outcome.code == 0, outcome.stderr
    lines = outcome.stdout.splitlines()
    assert lines[0] == "scrubbed:"
    assert lines[-1] != ""
    assert "" not in lines


def test_scrub_missing_trace_is_not_found(cli: Cli, bench_paths: BenchPaths) -> None:
    outcome = _run(cli, bench_paths, "nope", str(cli.root / "clean.jsonl"))

    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"


def test_scrub_unparsable_line_names_the_line(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    trace.write_text('{"seq": 1}\nnot json\n', encoding="utf-8")

    outcome = _run(cli, bench_paths, "t", str(cli.root / "clean.jsonl"))

    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "line 2" in str(outcome.error["message"])


def test_scrub_copies_entries_it_does_not_recognise(cli: Cli, bench_paths: BenchPaths) -> None:
    # `inspect` would reject this entry; scrub must still remove its secrets, so it works
    # on the raw documents rather than on validated entries.
    trace = bench_paths.traces_dir / "t.jsonl"
    trace.write_text(
        '{"seq": 1, "body": {"metadata": {"user_id": "op"}, "text": "/home/op/notes.md"},'
        ' "extra": [1, 2]}\n',
        encoding="utf-8",
    )
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    assert _lines(out) == [
        {"seq": 1, "body": {"text": "/home/user/notes.md"}, "extra": [1, 2]},
    ]


def test_scrub_bad_replacement_is_a_usage_error(cli: Cli, bench_paths: BenchPaths) -> None:
    trace = bench_paths.traces_dir / "t.jsonl"
    _fixture(trace)

    outcome = _run(cli, bench_paths, "t", str(cli.root / "clean.jsonl"), "--replace", "oops")

    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"


def test_scrub_resolves_the_packaged_sample(cli: Cli, bench_paths: BenchPaths) -> None:
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "sample-trace", str(out))

    assert outcome.code == 0, outcome.stderr
    assert outcome.document["entries_in"] == 30
    assert str(outcome.document["trace"]).endswith("sample-trace.jsonl.gz")
    assert len(_lines(out)) == 30


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_scrub_reports_the_payload_sizes(cli: Cli, bench_paths: BenchPaths, suffix: str) -> None:
    plain = bench_paths.traces_dir / "t.jsonl"
    _fixture(plain)
    trace = bench_paths.traces_dir / f"t{suffix}"
    if suffix.endswith(".gz"):
        write_trace_text(trace, read_trace_text(plain))
    out = cli.root / "clean.jsonl"

    outcome = _run(cli, bench_paths, "t", str(out))

    assert outcome.code == 0, outcome.stderr
    # Both sizes describe the uncompressed trace payload, so a .gz input is comparable.
    assert outcome.document["bytes_in"] == len(read_trace_text(trace).encode("utf-8"))
    assert outcome.document["bytes_out"] == out.stat().st_size


def test_gzip_output_has_no_timestamp_in_its_header() -> None:
    assert gzip.compress(b"", mtime=0)[4:8] == b"\x00\x00\x00\x00"
