"""Tests for promptreplay.bench.summary."""

from __future__ import annotations

import pytest

from promptreplay.bench.replay import ReplayResult
from promptreplay.bench.summary import sparkline, summarize
from promptreplay.bench.trace import Usage


def _result(
    turn: int,
    *,
    status: int = 200,
    provider: str | None = None,
    prompt_total: int = 0,
    cached: int = 0,
    latency_ms: float = 100.0,
) -> ReplayResult:
    return ReplayResult(
        seq=turn,
        turn=turn,
        status=status,
        latency_ms=latency_ms,
        provider=provider,
        usage=Usage(),
        prompt_total=prompt_total,
        cached=cached,
        cache_write=0,
        output_tokens=0,
    )


def test_sparkline_empty() -> None:
    assert sparkline([]) == ""


def test_sparkline_maps_range_to_blocks() -> None:
    s = sparkline([0.0, 0.5, 1.0])
    assert len(s) == 3
    assert s[0] == "▁"
    assert s[-1] == "█"


def test_sparkline_clamps_out_of_range_values() -> None:
    s = sparkline([-1.0, 2.0])
    assert s[0] == "▁"
    assert s[1] == "█"


def test_summarize_hit_ratio_none_when_fewer_than_two_successful() -> None:
    results = [_result(1, prompt_total=100, cached=0)]
    summary = summarize("label", results, [])
    assert summary.hit_ratio is None


def test_summarize_hit_ratio_over_turns_from_two_onward() -> None:
    results = [
        _result(1, prompt_total=100, cached=0),
        _result(2, prompt_total=100, cached=80),
        _result(3, prompt_total=100, cached=90),
    ]
    summary = summarize("label", results, [])
    # turn 1 excluded: cached 80+90=170 over prompt 100+100=200
    assert summary.hit_ratio == pytest.approx(0.85)


def test_summarize_drift_pinned_counts_turns_outside_requested_set() -> None:
    results = [
        _result(1, provider="Novita"),
        _result(2, provider="SiliconFlow"),
        _result(3, provider="Novita"),
    ]
    summary = summarize("label", results, ["novita"])
    assert summary.drift == 1


def test_summarize_drift_unpinned_counts_provider_changes() -> None:
    results = [
        _result(1, provider="Novita"),
        _result(2, provider="Novita"),
        _result(3, provider="SiliconFlow"),
        _result(4, provider="Novita"),
    ]
    summary = summarize("label", results, [])
    assert summary.drift == 2


def test_summarize_percentiles_and_curve_skip_error_turns_for_latency() -> None:
    results = [
        _result(1, status=200, latency_ms=100, prompt_total=100, cached=0),
        _result(2, status=200, latency_ms=200, prompt_total=100, cached=50),
        _result(3, status=0, latency_ms=999, prompt_total=0, cached=0),
    ]
    summary = summarize("label", results, [])
    assert summary.turns == 3
    assert summary.ok == 2
    assert summary.errors == 1
    assert summary.curve == [0.0, 0.5, 0.0]
    assert summary.latency_p50_ms == 100
    assert summary.latency_p95_ms == 200


def test_summarize_notes_are_deduplicated_in_order() -> None:
    r1 = _result(1)
    r1.note = "thinking param dropped"
    r2 = _result(2)
    r2.note = "thinking param dropped"
    r3 = _result(3)
    r3.note = "no price table for m"
    summary = summarize("label", [r1, r2, r3], [])
    assert summary.notes == ["thinking param dropped", "no price table for m"]
