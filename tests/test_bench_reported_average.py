"""Tests for `bench.reported_average`: applying the fetched figure, and the words for it."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from provibench.bench.openrouter_stats import ReportedAverage, ReportedShare
from provibench.bench.probe_summary import ProbeSummary
from provibench.bench.reported_average import (
    apply_reported_average,
    pooled_note,
    reported_average_block,
    reported_average_sentence,
)

_MODEL = "z-ai/glm-5.3-flash"


@dataclass(frozen=True)
class _Spec:
    """The minimum `probe_drift.SpecRef` shape: a target kind, the model it asked for, and
    the providers it pinned."""

    kind: str
    providers: list[str] = field(default_factory=list[str])
    model: str = _MODEL


def _summary(label: str, **fields: object) -> ProbeSummary:
    return ProbeSummary.model_validate({"label": label, **fields})


def _average(*, model: str = _MODEL, **shares: ReportedShare) -> ReportedAverage:
    return ReportedAverage(
        day="2026-09-20",
        model=model,
        permaslug="z-ai/glm-5.3-flash-20260826",
        fetched_at="x",
        shares=shares,
    )


_GMICLOUD_SHARE = ReportedShare(share_pct=69.6, endpoints=1, tokens=11815690336)


def test_apply_reported_average_computes_gmicloud_eff_price_from_listed_prices() -> None:
    summary = _summary("or:model@gmicloud/fp8", input_price=0.075, cache_read_price=0.015)
    spec = _Spec(kind="openrouter", providers=["gmicloud/fp8"])
    average = _average(gmicloud=_GMICLOUD_SHARE)

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=None)

    assert result.or_avg_share_pct == pytest.approx(69.6)
    assert result.or_avg_pooled is False
    assert result.or_avg_eff_per_m_prompt == pytest.approx(0.03324, abs=1e-6)


def test_vs_or_avg_pct_is_negative_when_this_run_cached_better() -> None:
    summary = _summary(
        "or:model@gmicloud/fp8",
        input_price=0.075,
        cache_read_price=0.015,
        eff_per_m_prompt=0.02,
    )
    spec = _Spec(kind="openrouter", providers=["gmicloud/fp8"])
    average = _average(gmicloud=_GMICLOUD_SHARE)

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=None)

    assert result.vs_or_avg_pct is not None
    assert result.vs_or_avg_pct < 0


def test_vs_or_avg_pct_is_positive_when_this_run_cached_worse() -> None:
    summary = _summary(
        "or:model@gmicloud/fp8",
        input_price=0.075,
        cache_read_price=0.015,
        eff_per_m_prompt=0.05,
    )
    spec = _Spec(kind="openrouter", providers=["gmicloud/fp8"])
    average = _average(gmicloud=_GMICLOUD_SHARE)

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=None)

    assert result.vs_or_avg_pct is not None
    assert result.vs_or_avg_pct > 0


def test_vs_or_avg_pct_uses_the_session_bill_when_the_trace_size_is_known() -> None:
    summary = _summary(
        "or:model@gmicloud/fp8",
        input_price=0.075,
        cache_read_price=0.015,
        eff_per_m_prompt=0.02,
        session_prompt_usd=0.02 * 1_800_000 / 1e6,
    )
    spec = _Spec(kind="openrouter", providers=["gmicloud/fp8"])
    average = _average(gmicloud=_GMICLOUD_SHARE)

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=1_800_000)

    assert result.or_avg_session_prompt_usd is not None
    assert result.or_avg_eff_per_m_prompt is not None
    # Same ratio either way: the session bill and the eff price differ by one constant factor.
    assert result.vs_or_avg_pct == pytest.approx((0.02 / result.or_avg_eff_per_m_prompt - 1) * 100)


def test_apply_reported_average_leaves_a_native_spec_untouched() -> None:
    summary = _summary("anthropic:model", input_price=0.075, cache_read_price=0.015)
    spec = _Spec(kind="anthropic", providers=[])
    average = _average(gmicloud=_GMICLOUD_SHARE)

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=None)

    assert result.or_avg_share_pct is None


def test_apply_reported_average_leaves_an_unpinned_openrouter_spec_untouched() -> None:
    summary = _summary("or:model", input_price=0.075, cache_read_price=0.015)
    spec = _Spec(kind="openrouter", providers=[])
    average = _average(gmicloud=_GMICLOUD_SHARE)

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=None)

    assert result.or_avg_share_pct is None


def test_apply_reported_average_leaves_a_spec_of_a_different_model_untouched() -> None:
    """Two models pinned to the same provider: the fetch was made for one of them, so the
    other's spec keeps none of the five fields even though its provider slug matches."""
    summary = _summary("or:other-model@gmicloud/fp8", input_price=0.075, cache_read_price=0.015)
    spec = _Spec(kind="openrouter", providers=["gmicloud/fp8"], model="other/model")
    average = _average(gmicloud=_GMICLOUD_SHARE)  # fetched for _MODEL, not "other/model"

    [result] = apply_reported_average([summary], [spec], average, trace_prompt_tokens=None)

    assert result.or_avg_share_pct is None


def test_apply_reported_average_returns_the_summaries_unchanged_without_an_average() -> None:
    summary = _summary("or:model@gmicloud/fp8", input_price=0.075, cache_read_price=0.015)
    spec = _Spec(kind="openrouter", providers=["gmicloud/fp8"])

    [result] = apply_reported_average([summary], [spec], None, trace_prompt_tokens=None)

    assert result.or_avg_share_pct is None


def _scored(label: str, *, h: float, or_avg: float) -> ProbeSummary:
    return _summary(label, h=h, or_avg_share_pct=or_avg)


def test_reported_average_sentence_names_the_endpoint_that_cached_worse() -> None:
    summaries = [
        _scored("@a", h=0.90, or_avg=80.0),
        _scored("@b", h=0.90, or_avg=80.0),
        _scored("@c", h=0.90, or_avg=80.0),
        _scored("@d", h=0.90, or_avg=80.0),
        _scored("@e", h=0.90, or_avg=80.0),
        _scored("@gmicloud/fp8", h=0.60, or_avg=70.0),
    ]
    labels = [s.label for s in summaries]

    sentence = reported_average_sentence(summaries, labels)

    assert sentence == (
        "This session's share sits above OpenRouter's average on five of the six endpoints "
        "and below it on @gmicloud/fp8."
    )


def test_reported_average_sentence_omits_the_below_clause_when_nothing_was_below() -> None:
    summaries = [_scored(f"@{name}", h=0.90, or_avg=80.0) for name in "abcdef"]
    labels = [s.label for s in summaries]

    sentence = reported_average_sentence(summaries, labels)

    assert (
        sentence
        == "This session's share sits above OpenRouter's average on six of the six endpoints."
    )


def test_reported_average_sentence_for_one_endpoint_below_names_it_alone() -> None:
    """A one-endpoint run that cached worse: no "0 of the one endpoints"."""
    summaries = [_scored("@z-ai", h=0.518, or_avg=91.3)]

    sentence = reported_average_sentence(summaries, ["@z-ai"])

    assert sentence == "This session's share sits below OpenRouter's average on @z-ai."


def test_reported_average_sentence_for_one_endpoint_above_says_the_one_endpoint() -> None:
    summaries = [_scored("@z-ai", h=0.95, or_avg=91.3)]

    sentence = reported_average_sentence(summaries, ["@z-ai"])

    assert sentence == "This session's share sits above OpenRouter's average on the one endpoint."


def test_reported_average_sentence_when_nothing_was_above_lists_only_the_below() -> None:
    summaries = [_scored("@a", h=0.60, or_avg=80.0), _scored("@b", h=0.60, or_avg=80.0)]

    sentence = reported_average_sentence(summaries, ["@a", "@b"])

    assert sentence == "This session's share sits below OpenRouter's average on @a and @b."


def test_reported_average_sentence_when_every_share_matches() -> None:
    summaries = [_scored("@a", h=0.80, or_avg=80.0), _scored("@b", h=0.80, or_avg=80.0)]

    sentence = reported_average_sentence(summaries, ["@a", "@b"])

    assert sentence == "This session's share matches OpenRouter's average on all two endpoints."


def test_reported_average_sentence_is_none_without_any_figure() -> None:
    summaries = [_summary("@a"), _summary("@b")]
    labels = [s.label for s in summaries]

    assert reported_average_sentence(summaries, labels) is None


def test_reported_average_block_marks_a_pooled_row_and_formats_the_vs_cell() -> None:
    summary = _summary(
        "@sail-research",
        h=0.80,
        eff_per_m_prompt=0.03,
        session_prompt_usd=0.06,
        or_avg_share_pct=92.7,
        or_avg_pooled=True,
        or_avg_eff_per_m_prompt=0.031,
        or_avg_session_prompt_usd=0.062,
        vs_or_avg_pct=0.2,
    )
    block = reported_average_block([summary], ["@sail-research"])

    assert block is not None
    assert block.columns == (
        "endpoint",
        "cache %",
        "OR avg",
        "eff $/M",
        "OR avg",
        "this trace $",
        "OR avg",
        "vs OR avg",
    )
    [row] = block.rows
    assert row[0] == "@sail-research"
    assert row[1] == "80.0"
    assert row[2] == "92.7*"
    assert row[3] == "0.030"
    assert row[4] == "0.031*"
    assert row[5] == "$0.0600"
    assert row[6] == "$0.0620"
    assert row[7] == "0%"  # |0.2| < 0.5 rounds to the "no change" cell


def test_reported_average_block_is_none_when_no_summary_has_a_figure() -> None:
    assert reported_average_block([_summary("@a")], ["@a"]) is None


def test_pooled_note_names_no_specific_endpoint() -> None:
    note = pooled_note()
    assert note.startswith("*")
    assert "provider" in note
