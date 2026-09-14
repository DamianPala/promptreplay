"""Tests for provibench.bench.trace."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest

from provibench.bench.trace import (
    PACKAGED_SAMPLE,
    TraceEntry,
    TraceNotFound,
    TraceParseError,
    load_trace,
    main_conversation,
    read_trace_text,
    resolve_trace,
    sample_trace_path,
    trace_name,
    write_trace_text,
)


def _entry(seq: int, conv: str, size: int) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        path="/v1/messages",
        body={"messages": [{"role": "user", "content": "x" * size}]},
        conversation=conv,
    )


def test_main_conversation_picks_group_with_most_bytes() -> None:
    entries = [
        _entry(1, "a", 10),
        _entry(2, "a", 10),
        _entry(3, "b", 1000),
    ]
    main = main_conversation(entries)
    assert len(main) == 1
    assert all(e.conversation == "b" for e in main)


def test_main_conversation_empty_trace() -> None:
    assert main_conversation([]) == []


@pytest.mark.parametrize("name", ["t.jsonl", "t.jsonl.gz", "t.gz", "t"])
def test_trace_name_strips_the_file_suffixes(name: str) -> None:
    assert trace_name(Path("/traces") / name) == "t"


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".gz"])
def test_a_trace_round_trips_through_every_suffix(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"t{suffix}"
    entry = _entry(1, "c1", 10)

    write_trace_text(path, entry.model_dump_json(exclude_none=True) + "\n")

    assert load_trace(path) == [entry]
    assert "c1" in read_trace_text(path)
    if suffix == ".gz":
        assert path.read_bytes()[:2] == b"\x1f\x8b"


def test_load_trace_names_the_line_it_cannot_parse(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    first = _entry(1, "c1", 10).model_dump_json(exclude_none=True)
    path.write_text(f"{first}\nnot json\n", encoding="utf-8")

    with pytest.raises(TraceParseError) as raised:
        load_trace(path)

    assert raised.value.line == 2
    assert "line 2" in str(raised.value)


def test_load_trace_names_the_field_that_is_not_an_entry(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text('{"seq": 1, "ts": "t", "path": "/v1/messages", "body": {}}\n', encoding="utf-8")

    with pytest.raises(TraceParseError) as raised:
        load_trace(path)

    assert raised.value.line == 1
    assert "conversation" in str(raised.value)


def test_resolve_trace_prefers_an_existing_path(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    explicit = tmp_path / "elsewhere.jsonl"
    explicit.touch()

    assert resolve_trace(str(explicit), traces_dir, tmp_path) == explicit


@pytest.mark.parametrize("name", ["t", "t.jsonl", "t.jsonl.gz"])
def test_resolve_trace_finds_a_name_under_the_traces_directory(tmp_path: Path, name: str) -> None:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    (traces_dir / name).touch()

    assert resolve_trace(name, traces_dir, tmp_path) == traces_dir / name


def test_resolve_trace_reports_an_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(TraceNotFound):
        resolve_trace("nope", tmp_path / "traces", tmp_path)


def test_resolve_trace_finds_the_packaged_sample_last(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    assert resolve_trace("sample", traces_dir, tmp_path) == sample_trace_path()
    assert sample_trace_path().name == PACKAGED_SAMPLE

    # A trace of the caller's own named `sample` wins over the packaged example.
    own = traces_dir / "sample.jsonl"
    own.touch()
    assert resolve_trace("sample", traces_dir, tmp_path) == own


def test_the_packaged_sample_is_a_scrubbed_thirty_turn_session() -> None:
    """The shipped sample: 30 turns of one Claude Code session, scrubbed before packaging."""
    path = sample_trace_path()

    entries = load_trace(path)
    assert [entry.seq for entry in entries] == list(range(1, 31))
    assert len({entry.conversation for entry in entries}) == 1
    assert all(entry.response is not None for entry in entries)
    assert all("metadata" not in entry.body for entry in entries)
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    assert re.search(r"/home/(?!user\b)\w+", text) is None
    assert re.search(r"-home-(?!user-)\w+", text) is None
    assert "user_id" not in text
