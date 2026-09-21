"""Cost breakdown: input / cache read / cache write / output, in USD."""

from __future__ import annotations

from pydantic import BaseModel

from provibench.bench.targets import Prices


class CostBreakdown(BaseModel):
    input: float
    cache_read: float
    cache_write: float
    output: float
    source: str

    @property
    def total(self) -> float:
        return self.input + self.cache_read + self.cache_write + self.output

    def __add__(self, other: object) -> CostBreakdown:
        if not isinstance(other, CostBreakdown):
            return NotImplemented
        source = self.source if self.source == other.source else "mixed"
        return CostBreakdown(
            input=self.input + other.input,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            output=self.output + other.output,
            source=source,
        )


def compute_cost(
    *,
    input_tokens: int,
    cache_read: int,
    cache_write: int,
    output_tokens: int,
    prices: Prices,
    source: str,
) -> CostBreakdown:
    scale = 1e-6
    return CostBreakdown(
        input=input_tokens * prices.input * scale,
        cache_read=cache_read * prices.cache_read * scale,
        cache_write=cache_write * prices.cache_write * scale,
        output=output_tokens * prices.output * scale,
        source=source,
    )


def effective_price(h: float | None, prices: Prices | None) -> float | None:
    """USD per 1M prompt tokens at the given hit-weighted `h`: misses at input, hits at cache
    read. Shared by `probe_summary.summarize_probe` (this run's own `h`) and
    `reported_average` (OpenRouter's reported share in place of it), so the two never price
    the same pair of listed rates by two different formulas."""
    if h is None or prices is None:
        return None
    return (1 - h) * prices.input + h * prices.cache_read
