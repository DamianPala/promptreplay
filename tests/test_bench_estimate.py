"""Tests for the worst-case estimates: what they price, and what they say about it."""

from __future__ import annotations

from typing import Literal

import pytest

from provibench.bench.estimate import (
    PreCheckCost,
    SpecEstimate,
    SpecPrices,
    UpperBound,
    estimate_total,
    probe_estimate,
    render_estimate,
    replay_estimate,
    same_amount,
    spec_prices,
)
from provibench.bench.openrouter import Endpoint
from provibench.bench.probe_models import ProbeOptions
from provibench.bench.targets import Prices, RunSpec, Target
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage

_TABLE = Prices(input=1.0, cache_read=0.1, cache_write=1.0, output=2.0)
"""The table every priced estimate uses; output is twice input, so the split is visible."""
_STREAM = 256
"""The throughput request's output budget, which one rung of a probe estimate carries."""


def _options(
    rungs: list[int], repeats: list[int], *, throughput: bool = True, ttl_s: list[int] | None = None
) -> ProbeOptions:
    return ProbeOptions(
        rungs=rungs, repeats=repeats, gap_s=0.0, throughput=throughput, ttl_s=ttl_s
    ).resolved([_entry(seq, 100) for seq in range(1, 9)])


def _target(
    kind: Literal["openrouter", "anthropic"] = "anthropic",
    prices: dict[str, Prices] | None = None,
) -> Target:
    return Target(
        name="t",
        url="https://api.test/v1/messages",
        api_key_env="X_KEY",
        kind=kind,
        prices=prices or {},
    )


def _endpoint(tag: str, provider_name: str, prompt_per_token: str) -> Endpoint:
    return Endpoint(
        tag=tag,
        provider_name=provider_name,
        prices=Prices(
            input=float(prompt_per_token) * 1e6,
            cache_read=float(prompt_per_token) / 10 * 1e6,
            cache_write=float(prompt_per_token) * 1e6,
            output=float(prompt_per_token) * 2e6,
        ),
    )


def _entry(seq: int, prompt_tokens: int | None) -> TraceEntry:
    response = None
    if prompt_tokens is not None:
        response = RecordedResponse(
            status=200, latency_ms=1.0, usage=Usage(input_tokens=prompt_tokens)
        )
    return TraceEntry(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        path="/v1/messages",
        body={"messages": [{"role": "user", "content": f"question {seq}"}]},
        conversation="c1",
        response=response,
    )


# --- pricing ------------------------------------------------------------------


def test_spec_prices_takes_the_endpoint_a_pinned_spec_named() -> None:
    index = {
        "model": [
            _endpoint("cheap", "Cheap", "0.0000001"),
            _endpoint("novita", "Novita", "0.0000003"),
        ]
    }
    spec = RunSpec(target=_target("openrouter"), model="model", providers=["Novita"])
    prices = spec_prices(spec, index)
    assert prices is not None
    assert prices.prices.input == 0.3
    assert prices.source == "openrouter-endpoint"
    assert prices.provider == "Novita"


def test_spec_prices_takes_the_most_expensive_endpoint_when_unpinned() -> None:
    index = {
        "model": [
            _endpoint("cheap", "Cheap", "0.0000001"),
            _endpoint("pricey", "Pricey", "0.0000009"),
            _endpoint("mid", "Mid", "0.0000005"),
        ]
    }
    spec = RunSpec(target=_target("openrouter"), model="model")
    prices = spec_prices(spec, index)
    assert prices is not None
    assert prices.prices.input == pytest.approx(0.9)  # worst case, not the cheapest endpoint
    assert prices.provider == "Pricey"


def test_spec_prices_without_a_match_is_none() -> None:
    spec = RunSpec(target=_target("openrouter"), model="model", providers=["ghost"])
    assert spec_prices(spec, {"model": [_endpoint("cheap", "Cheap", "0.0000001")]}) is None
    assert spec_prices(spec, {}) is None
    assert spec_prices(RunSpec(target=_target(), model="model"), {}) is None


def test_spec_prices_reads_the_targets_table_for_a_native_spec() -> None:
    spec = RunSpec(target=_target("anthropic", {"model": _TABLE}), model="model")
    prices = spec_prices(spec, {})
    assert prices is not None
    assert prices.source == "targets"
    assert prices.provider is None


# --- estimates ----------------------------------------------------------------


def test_probe_estimate_counts_the_cold_write_and_every_warm_read() -> None:
    entries = [_entry(1, 100), _entry(2, 200), _entry(3, 300)]
    prices = SpecPrices(prices=_TABLE, source="targets")
    # rung 1: cold turn 1 (100) + 2 warm reads of turn 2 (2 * 200) + the streamed turn 2
    estimate = probe_estimate(_spec_anthropic(), entries, _options([1], [2]), prices)
    assert estimate.tokens == 100 + 2 * 200 + 200
    # the streamed request's own 256 output tokens are billed at the output price, 2.0
    assert estimate.usd == pytest.approx((700 * 1.0 + _STREAM * 2.0) * 1e-6)
    assert estimate.tokens_known is True


def test_probe_estimate_without_throughput_leaves_the_stream_out() -> None:
    entries = [_entry(1, 100), _entry(2, 200)]
    prices = SpecPrices(prices=_TABLE, source="targets")
    estimate = probe_estimate(
        _spec_anthropic(), entries, _options([1], [1], throughput=False), prices
    )
    assert estimate.tokens == 100 + 200
    assert estimate.usd == pytest.approx(300 * 1.0 * 1e-6)


def test_probe_estimate_counts_the_ttl_reads_of_the_first_rung_only() -> None:
    entries = [_entry(1, 100), _entry(2, 200), _entry(3, 300)]
    prices = SpecPrices(prices=_TABLE, source="targets")
    # an explicit rung list keeps `ttl_s` from tripping the gap check at the default 1 s
    estimate = probe_estimate(
        _spec_anthropic(),
        entries,
        _options([1], [1], throughput=False, ttl_s=[1, 2]),
        prices,
    )
    assert estimate.tokens == 100 + 200 + 2 * 200  # the cold, the warm read, two re-reads


def test_probe_estimate_marks_unknown_tokens_and_prices_nothing() -> None:
    entries = [_entry(1, None), _entry(2, 200)]
    prices = SpecPrices(prices=_TABLE, source="targets")
    estimate = probe_estimate(_spec_anthropic(), entries, _options([1], [1]), prices)
    assert estimate.tokens == 200 + 200  # turn 1's unknown size is not counted
    assert estimate.tokens_known is False
    assert estimate.usd is None  # a total over unknown tokens would be a guess
    assert "turn 1 has no recorded prompt usage; its tokens are unknown" in estimate.notes


def test_probe_estimate_notes_that_an_unpinned_endpoint_is_the_expensive_one() -> None:
    entries = [_entry(1, 100), _entry(2, 200)]
    index = {
        "model": [
            _endpoint("cheap", "Cheap", "0.0000001"),
            _endpoint("pricey", "Pricey", "0.0000009"),
        ]
    }
    spec = RunSpec(target=_target("openrouter"), model="model")
    prices = spec_prices(spec, index)
    estimate = probe_estimate(spec, entries, _options([1], [1]), prices)
    assert estimate.usd is not None
    [note] = estimate.notes
    assert "most expensive endpoint (Pricey)" in note
    assert "@provider" in note


def test_probe_estimate_without_a_price_is_tokens_only() -> None:
    entries = [_entry(1, 100), _entry(2, 200)]
    estimate = probe_estimate(_spec_anthropic(), entries, _options([1], [1]), None)
    assert estimate.tokens == 100 + 200 + 200
    assert estimate.usd is None
    assert estimate.notes == ["no listed input price for this model; the estimate is tokens only"]


def test_replay_estimate_counts_every_selected_turn_once() -> None:
    entries = [_entry(1, 100), _entry(2, 200), _entry(3, 300)]
    prices = SpecPrices(prices=_TABLE, source="targets")
    estimate = replay_estimate(_spec_anthropic(), entries, prices)
    assert estimate.tokens == 600
    assert estimate.usd == 600 * 1.0 * 1e-6


def _spec_anthropic() -> RunSpec:
    return RunSpec(target=_target("anthropic", {"model": _TABLE}), model="model")


# --- rendering ----------------------------------------------------------------


def test_estimate_total_sums_what_is_known() -> None:
    entries = [_entry(1, 100), _entry(2, 100)]
    table = SpecPrices(prices=_TABLE, source="targets")
    priced = probe_estimate(_spec_anthropic(), entries, _options([1], [0], throughput=False), table)
    unpriced = probe_estimate(
        _spec_anthropic(), entries, _options([1], [0], throughput=False), None
    )
    assert estimate_total([priced]) == 100 * 1.0 * 1e-6
    assert estimate_total([priced, unpriced]) == 100 * 1.0 * 1e-6  # the known part only
    assert estimate_total([unpriced]) is None


def test_render_estimate_says_retries_are_excluded() -> None:
    entries = [_entry(1, 100), _entry(2, 100)]
    estimate = probe_estimate(
        _spec_anthropic(),
        entries,
        _options([1], [0], throughput=False),
        SpecPrices(prices=_TABLE, source="targets"),
    )
    text = render_estimate([estimate])
    assert "excluding retries" in text
    assert "worst case assumes no cache hit" in text


def test_render_estimate_without_specs_is_only_a_header_and_a_total() -> None:
    lines = render_estimate([]).splitlines()
    assert lines[0].startswith("endpoint")
    assert lines[2].startswith("total")


def _priced(label: str) -> SpecEstimate:
    return SpecEstimate(
        label=label, tokens=100, prices=SpecPrices(prices=_TABLE, source="targets"), usd=0.0001
    )


def test_render_estimate_shortens_the_spec_column_like_the_probe_tables() -> None:
    """A sweep's estimate is read for its endpoints, so the endpoints have to be readable."""
    text = render_estimate(
        [
            _priced("openrouter:deepseek/deepseek-v4.1-flash@relace/fp8"),
            _priced("openrouter:deepseek/deepseek-v4.1-flash@deepinfra/turbo"),
            _priced("deepseek:deepseek-flash"),
        ]
    )
    lines = text.splitlines()
    assert lines[0] == "endpoints: openrouter:deepseek/deepseek-v4.1-flash@<provider>"
    assert lines[1].startswith("endpoint")  # the table itself is unchanged, one line lower
    assert "@relace/fp8" in text and "@deepinfra/turbo" in text
    assert "deepseek:deepseek-flash" in text
    assert "…" not in text  # nothing had to be cut


def test_render_estimate_elides_a_lone_long_label_keeping_its_provider() -> None:
    text = render_estimate([_priced(f"openrouter:{'deepseek/' + 'x' * 60}@novita")])
    assert "…" in text
    assert "@novita" in text  # the part that names the row survives the cut


def test_render_estimate_prints_the_upper_bound_under_the_check() -> None:
    """The rows stay the ranked expectation; the line under the check prices the priciest cut."""
    estimates = [_priced("fake:a"), _priced("fake:b")]
    cost = PreCheckCost(requests=2, tokens=200, usd=0.0002)
    lines = render_estimate(
        estimates, pre_check=cost, upper_bound=UpperBound(keep=2, usd=0.0009)
    ).splitlines()
    check = next(index for index, line in enumerate(lines) if line.startswith("pre-check: "))
    assert lines[check + 1] == (
        "upper bound if the 2 priciest candidates are the ones that answer: $0.0009"
    )
    # the total is already 0.0004, so a bound equal to it says nothing the table does not
    same = render_estimate(estimates, pre_check=cost, upper_bound=UpperBound(keep=2, usd=0.0004))
    assert "upper bound" not in same


def test_same_amount_tolerates_a_reordered_sum() -> None:
    assert same_amount(0.1 + 0.2, 0.3)
    assert same_amount(None, None)
    assert not same_amount(None, 1.0) and not same_amount(1.0, None)
    assert not same_amount(1.0, 1.001)
