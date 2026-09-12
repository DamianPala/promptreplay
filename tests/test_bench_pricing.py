"""Tests for provibench.bench.pricing."""

from __future__ import annotations

from provibench.bench.pricing import CostBreakdown, compute_cost
from provibench.bench.targets import Prices


def test_compute_cost_basic() -> None:
    prices = Prices(input=1.0, cache_read=0.5, cache_write=2.0, output=4.0)
    cost = compute_cost(
        input_tokens=1_000_000,
        cache_read=1_000_000,
        cache_write=500_000,
        output_tokens=250_000,
        prices=prices,
        source="table",
    )
    assert cost.input == 1.0
    assert cost.cache_read == 0.5
    assert cost.cache_write == 1.0
    assert cost.output == 1.0
    assert cost.total == 3.5
    assert cost.source == "table"


def test_compute_cost_zero_tokens() -> None:
    prices = Prices(input=1, cache_read=1, cache_write=1, output=1)
    cost = compute_cost(
        input_tokens=0, cache_read=0, cache_write=0, output_tokens=0, prices=prices, source="table"
    )
    assert cost.total == 0.0


def test_cost_breakdown_add_same_source_keeps_source() -> None:
    a = CostBreakdown(input=1, cache_read=1, cache_write=1, output=1, source="table")
    b = CostBreakdown(input=2, cache_read=2, cache_write=2, output=2, source="table")
    c = a + b
    assert c.total == 12
    assert c.source == "table"


def test_cost_breakdown_add_mixed_source() -> None:
    a = CostBreakdown(input=1, cache_read=0, cache_write=0, output=0, source="table")
    b = CostBreakdown(input=1, cache_read=0, cache_write=0, output=0, source="openrouter-endpoint")
    c = a + b
    assert c.source == "mixed"
    assert c.total == 2
