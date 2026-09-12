"""Tests for provibench.bench.trace."""

from __future__ import annotations

from provibench.bench.trace import TraceEntry, main_conversation


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
