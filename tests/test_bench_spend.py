"""Tests for promptreplay.bench.spend: what a probe run spent, billed or priced by hand."""

from __future__ import annotations

import pytest

from promptreplay.bench.estimate import SpecPrices
from promptreplay.bench.probe_models import ProbeResult
from promptreplay.bench.spend import (
    precheck_spend,
    record_cost,
    spec_spend,
    total_spend,
    usage_cost,
    worst_case_usd,
)
from promptreplay.bench.targets import Prices


def _prices() -> SpecPrices:
    return SpecPrices(
        prices=Prices(input=1.0, cache_read=0.1, cache_write=1.0, output=2.0), source="table"
    )


def _record(**overrides: object) -> ProbeResult:
    base: dict[str, object] = {
        "spec_label": "fake:model-a",
        "rung": 1,
        "role": "warm",
        "attempt": 1,
        "seq": 1,
        "status": 200,
        "latency_ms": 10.0,
        "prompt_total": 100,
        "cached": 0,
        "cache_write": 0,
    }
    base.update(overrides)
    return ProbeResult.model_validate(base)


def test_record_cost_prices_uncached_cached_written_and_output_tokens() -> None:
    """The three prompt buckets are disjoint: `prompt_total` already contains the cached and
    the written tokens, so each is billed at its own rate and never also at the input rate."""
    record = _record(prompt_total=150, cached=50, cache_write=20, usage={"output_tokens": 10})
    cost = record_cost(record, _prices())
    # 80 uncached @1.0 + 50 cached @0.1 + 20 written @1.0 + 10 output @2.0, per 1M
    assert cost == pytest.approx((80 * 1.0 + 50 * 0.1 + 20 * 1.0 + 10 * 2.0) * 1e-6)


def test_record_cost_is_none_without_a_listed_price() -> None:
    assert record_cost(_record(), None) is None


def test_spec_spend_prefers_billed_usd_over_the_usage_estimate() -> None:
    """Item 3: OpenRouter's own bill wins when there is one; a native spec never gets one,
    so it falls back to the usage priced at the run's own listed rates."""
    records = [_record(cached=50)]
    assert spec_spend(records, billed_usd=0.5, prices=_prices()) == 0.5
    assert spec_spend(records, billed_usd=None, prices=_prices()) == pytest.approx(
        usage_cost(records, _prices())
    )


def test_spec_spend_is_none_for_a_native_spec_with_no_listed_price() -> None:
    assert spec_spend([_record()], billed_usd=None, prices=None) is None


def test_total_spend_sums_only_the_known_amounts() -> None:
    assert total_spend([0.1, None, 0.2]) == pytest.approx(0.3)
    assert total_spend([None, None]) is None


def test_precheck_spend_prices_each_candidate_at_its_own_listed_rate() -> None:
    records = [
        _record(spec_label="a", prompt_total=100),
        _record(spec_label="b", prompt_total=200),
    ]
    prices = {"a": _prices(), "b": _prices()}
    assert precheck_spend(records, prices) == pytest.approx((100 + 200) * 1.0 * 1e-6)


def test_precheck_spend_ignores_a_candidate_with_no_listed_price() -> None:
    records = [_record(spec_label="a", prompt_total=100), _record(spec_label="b")]
    assert precheck_spend(records, {"a": _prices()}) == pytest.approx(100 * 1.0 * 1e-6)


def test_worst_case_usd_prices_every_prompt_token_at_the_input_rate() -> None:
    """The estimate's own worst case, reconstructed after the fact: no cache hit, every
    served prompt token at the listed input price, ignoring what actually happened to it."""
    records = [_record(prompt_total=100, cached=90), _record(prompt_total=200, cached=0)]
    assert worst_case_usd(records, _prices()) == pytest.approx((100 + 200) * 1.0 * 1e-6)


def test_worst_case_usd_is_none_without_a_listed_price_or_a_served_record() -> None:
    assert worst_case_usd([_record()], None) is None
    assert worst_case_usd([_record(status=500, error="boom")], _prices()) is None
