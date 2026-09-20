"""What a rung's own first read reveals, and what its reads bill beyond their tokens.

An agent loop only ever performs a rung's first warm read: every later repeat re-reads the
prompt the first one just wrote, so pooling every repeat the way `ProbeSummary.hit_rate`
does flatters every provider's cache. This module pools only the first reads the same way,
and separately counts the output tokens a `max_tokens: 1` read was never supposed to produce
and the requests a run's own pace turned into a 429 -- both folded into `ProbeSummary` next
to it. A leaf module: it does not import `probe_summary`, so `probe_summary` can import it.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from provibench.bench.probe_models import ProbeResult, is_served

__all__ = ["first_read_stats", "output_token_stats", "output_tokens_note", "rate_limited_count"]

_MAX_TOKENS_PER_READ = 1
_HTTP_TOO_MANY_REQUESTS = 429
_READ_ROLES = ("cold", "warm", "ttl")
"""Requests sent with `max_tokens: 1`; the streamed throughput request is not one of them."""


def first_read_stats(
    records: Sequence[ProbeResult],
) -> tuple[float | None, float | None, float | None]:
    """`(first_hit_rate, first_cached_fraction, first_h)`, pooled over every rung's read 1."""
    prefixes = {rung: _cold_prefix(records, rung) for rung in {r.rung for r in records}}
    first_reads = [r for r in records if r.role == "warm" and r.attempt == 1]
    served = [r for r in first_reads if is_served(r)]
    hits = [r for r in served if _cached(r) > 0]
    hit_rate = len(hits) / len(served) if served else None
    cached_fraction = fmean(_fraction(_cached(r), prefixes[r.rung]) for r in hits) if hits else None
    h = hit_rate * (cached_fraction or 0.0) if hit_rate is not None else None
    return hit_rate, cached_fraction, h


def output_token_stats(records: Sequence[ProbeResult]) -> tuple[int, int]:
    """`(output_tokens, reads)`: total output tokens and how many one-token reads served them."""
    reads = [record for record in records if record.role in _READ_ROLES and is_served(record)]
    return sum(record.usage.output_tokens for record in reads), len(reads)


def output_tokens_note(output_tokens: int, reads: int) -> str | None:
    """The note when reads returned more than their `max_tokens: 1` budget, else `None`.

    Every renderer already prefixes a note with its endpoint's own label (the text report's
    `label: note`, the HTML page's bold short label), so the sentence itself names what
    happened and what it costs without repeating who did it.
    """
    if reads == 0 or output_tokens <= reads * _MAX_TOKENS_PER_READ:
        return None
    return (
        f"ignored the one-token limit on the cache probes and generated {output_tokens:,} "
        "tokens. That raised this run's cost, not the prices above."
    )


def rate_limited_count(records: Sequence[ProbeResult]) -> int:
    """How many of a spec's requests came back HTTP 429: the run's own pace, not a refusal."""
    return sum(1 for record in records if record.status == _HTTP_TOO_MANY_REQUESTS)


def _cold_prefix(records: Sequence[ProbeResult], rung: int) -> int:
    for record in records:
        if record.rung == rung and record.role == "cold":
            return record.prompt_total
    return 0


def _cached(record: ProbeResult) -> int:
    """Mirrors `probe_summary.cached_of` without importing it, to keep this module a leaf."""
    if record.cached > 0:
        return record.cached
    return record.native_tokens_cached or 0


def _fraction(cached: int, prefix: int) -> float:
    if prefix <= 0:
        return 0.0
    return min(cached / prefix, 1.0)
