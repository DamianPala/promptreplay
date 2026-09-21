"""One rung's cold write and warm reads, folded into `RungSummary`.

Split out of `bench.probe_summary` purely for that module's line budget: a rung's own
numbers -- its hit rate, its TTL re-reads, its cold/warm latency -- are one self-contained
step of `summarize_probe`, and moving them here is what leaves room for a spec-level field
without cutting anything a reader depends on. A leaf module: it imports only `probe_models`,
so `probe_summary` (and anything else) can import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median

from pydantic import AliasChoices, BaseModel, Field

from promptreplay.bench.probe_models import ProbeResult, is_failed, is_served

__all__ = ["RungSummary", "TtlRead", "cached_of", "summarize_rung"]

_HTTP_TOO_MANY_REQUESTS = 429


class TtlRead(BaseModel):
    """One `--ttl` re-read: how long after the warm reads, and what came back."""

    offset_s: int = Field(validation_alias=AliasChoices("offset_s", "offset"))
    hit: bool
    fraction: float | None = None
    offset_actual_s: float | None = None
    """When the read really went out: seconds past the warm baseline, or `None` if unknown."""

    @property
    def lateness_s(self) -> float:
        """How much later than asked for the read went out."""
        return (
            0.0 if self.offset_actual_s is None else max(self.offset_actual_s - self.offset_s, 0.0)
        )


class RungSummary(BaseModel):
    """One rung's cold write, its warm reads and its extra requests, folded together."""

    endpoint: str
    rung: int
    prompt_cold: int
    cached_cold: int
    hit_rate: float
    cached_fraction: float | None = None
    hits: list[float | None] = Field(default_factory=list[float | None])
    """One entry per warm attempt: `None` failed, `0.0` missed, else the cached fraction."""
    cold_ms: float = 0.0
    warm_ms: float | None = None
    ttft_ms: float | None = None
    """The throughput request's time to first token; `None` when it did not run."""
    gen_tok_s: float | None = None
    fingerprint: str | None = None
    ttl: list[TtlRead] = Field(default_factory=list[TtlRead])
    cache_write_cold: int = 0
    errors: int = 0
    rate_limited: int = 0
    """Of `errors`, how many were an HTTP 429 -- the run's own rate limit, not a refusal."""
    retries: int = 0
    skipped: bool = False
    cold_error: str | None = None


def cached_of(record: ProbeResult | None) -> int:
    """A record's cached tokens: what the provider billed for, else the gateway's count.

    Some gateways report nothing about the cache in the response while their `/generation`
    endpoint carries the native count; this fallback is what keeps those providers from
    scoring zero, and `summarize_probe` notes when it was used.
    """
    if record is None:
        return 0
    if record.cached > 0:
        return record.cached
    return record.native_tokens_cached or 0


def summarize_rung(endpoint: str, rung: int, records: Sequence[ProbeResult]) -> RungSummary:
    """Fold one rung's requests into hit rate, fractions, timings and the TTL reads.

    A rung whose cold request was not served is `skipped`: it scores nothing — no fraction,
    an empty hit list — and carries the cold error, because a failed write is not a miss.
    """
    cold = next((record for record in records if record.role == "cold"), None)
    warm = [record for record in records if record.role == "warm"]
    served = [record for record in warm if is_served(record)]
    hits = [record for record in served if cached_of(record) > 0]
    prefix = cold.prompt_total if cold is not None else 0
    latencies = [record.latency_ms for record in served]
    stream = next((record for record in records if record.role == "stream"), None)
    return RungSummary(
        endpoint=endpoint,
        rung=rung,
        prompt_cold=prefix,
        cached_cold=cached_of(cold),
        hit_rate=len(hits) / len(served) if served else 0.0,
        cached_fraction=fmean(_fraction(cached_of(r), prefix) for r in hits) if hits else None,
        hits=[_hit(record, prefix) for record in warm],
        cold_ms=cold.latency_ms if cold is not None else 0.0,
        warm_ms=median(latencies) if latencies else None,
        ttft_ms=_stream_ttft(stream),
        gen_tok_s=stream.gen_tok_s if stream is not None else None,
        fingerprint=stream.fingerprint if stream is not None else None,
        ttl=[_ttl_read(record, prefix) for record in records if record.role == "ttl"],
        cache_write_cold=cold.cache_write if cold is not None else 0,
        errors=sum(1 for record in records if is_failed(record)),
        rate_limited=_rate_limited_count(records),
        retries=sum(record.retries for record in records),
        skipped=cold is None or not is_served(cold),
        cold_error=cold.error if cold is not None else None,
    )


def _stream_ttft(record: ProbeResult | None) -> float | None:
    return record.ttft_ms if record is not None and is_served(record) else None


def _ttl_read(record: ProbeResult, prefix: int) -> TtlRead:
    return TtlRead(
        offset_s=record.attempt,
        hit=is_served(record) and cached_of(record) > 0,
        fraction=_fraction(cached_of(record), prefix) if is_served(record) else None,
        offset_actual_s=record.offset_actual_s,
    )


def _fraction(cached: int, prefix: int) -> float:
    if prefix <= 0:
        return 0.0
    return min(cached / prefix, 1.0)


def _hit(record: ProbeResult, prefix: int) -> float | None:
    """One warm attempt's cache fraction, or `None` when the attempt failed."""
    if not is_served(record):
        return None
    return round(_fraction(cached_of(record), prefix), 2)


def _rate_limited_count(records: Sequence[ProbeResult]) -> int:
    """Mirrors `probe_reads.rate_limited_count` without importing it, to keep this a leaf."""
    return sum(1 for record in records if record.status == _HTTP_TOO_MANY_REQUESTS)
