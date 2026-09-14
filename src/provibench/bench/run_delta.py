"""Two runs subtracted spec by spec: what each metric was, and how much it moved.

`bench/run_history` reads a runs directory as a time series; this is the other half of the
time axis, the arithmetic of `compare`. Every metric is a pair — the value in the earlier
run and in the later one — and the delta between them, which is `None` whenever either
side has no number to subtract. That is deliberate: a spec that measured nothing in one of
the two runs did not measure zero, and printing a delta against zero would invent a
collapse the run never reported.

The rows come out in the order the earlier run measured them. A sweep orders its own rows
by the effective price it measured, and that order is the verdict a comparison is read
against: a spec whose price rose is spotted by where it sat in the first run, not by an
alphabetical column.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from provibench.bench.run_history import RunNumbers, RunRow

__all__ = ["Comparison", "MetricPair", "PriceDelta", "SpecDelta", "compare_runs"]


class MetricPair(BaseModel):
    """One metric in both runs of a comparison, and the change between them."""

    a: float | None = None
    b: float | None = None

    @property
    def delta(self) -> float | None:
        """`b - a`, or `None` when either side has no number to subtract."""
        if self.a is None or self.b is None:
            return None
        return self.b - self.a


class SpecDelta(BaseModel):
    """One spec present in both runs: its numbers there and here."""

    spec: str
    hit_rate: MetricPair = Field(default_factory=MetricPair)
    eff_per_m_prompt: MetricPair = Field(default_factory=MetricPair)
    ttft_ms: MetricPair = Field(default_factory=MetricPair)
    gen_tok_s: MetricPair = Field(default_factory=MetricPair)


class PriceDelta(BaseModel):
    """One spec whose endpoint both runs' snapshots priced: the two prompt-side prices.

    A cache benchmark is about the cache-read price as much as the input one — a provider
    that raises its cache-read rate changes what a warm run costs — so both are carried.
    """

    spec: str
    price_in: MetricPair = Field(default_factory=MetricPair)
    price_cache_read: MetricPair = Field(default_factory=MetricPair)


class Comparison(BaseModel):
    """Two runs of one trace and protocol, and every spec either of them measured."""

    a: RunNumbers
    b: RunNumbers
    rows: list[SpecDelta] = Field(default_factory=list[SpecDelta])
    listed: list[PriceDelta] = Field(default_factory=list[PriceDelta])
    only_in_a: list[str] = Field(default_factory=list[str])
    only_in_b: list[str] = Field(default_factory=list[str])


def compare_runs(a: RunNumbers, b: RunNumbers) -> Comparison:
    """The delta per spec of two runs, in the order the first of them measured them.

    A spec only one of the runs measured cannot be subtracted at all, so it is named apart
    instead of paired, and so is a spec one of the snapshots never priced.
    """
    here = {row.spec: row for row in b.rows}
    pairs = [(row, here[row.spec]) for row in a.rows if row.spec in here]
    return Comparison(
        a=a,
        b=b,
        rows=[_delta(earlier, later) for earlier, later in pairs],
        listed=[
            PriceDelta(
                spec=earlier.spec,
                price_in=MetricPair(a=earlier.listed_input, b=later.listed_input),
                price_cache_read=MetricPair(a=earlier.listed_cache_read, b=later.listed_cache_read),
            )
            for earlier, later in pairs
            if earlier.listed_input is not None and later.listed_input is not None
        ],
        only_in_a=[row.spec for row in a.rows if row.spec not in here],
        only_in_b=[row.spec for row in b.rows if row.spec not in {row.spec for row in a.rows}],
    )


def _delta(earlier: RunRow, later: RunRow) -> SpecDelta:
    return SpecDelta(
        spec=earlier.spec,
        hit_rate=MetricPair(a=earlier.hit_rate, b=later.hit_rate),
        eff_per_m_prompt=MetricPair(a=earlier.eff_per_m_prompt, b=later.eff_per_m_prompt),
        ttft_ms=MetricPair(a=earlier.ttft_ms, b=later.ttft_ms),
        gen_tok_s=MetricPair(a=earlier.gen_tok_s, b=later.gen_tok_s),
    )
