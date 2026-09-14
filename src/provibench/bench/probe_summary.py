"""ProbeSummary: a probe run's records folded into the numbers the probe exists for.

Hit rate is hits ÷ served warm attempts — not the mean of the per-hit fractions, which
hides a skipped rung and flatters a run that never hit. The prefix fraction says how much
of the cold prefix each hit really covered, and `h` — the two multiplied — is what the
effective input price is weighted by. Both are pure functions of the records, so `report`
rebuilds them from a run directory with no network.

The other question the probe exists for — which endpoint is fast, and which one is really
serving the model — is answered by the fields here and the comparisons in `probe_drift`.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median

from pydantic import BaseModel, Field

from provibench.bench.estimate import SpecPrices
from provibench.bench.probe_models import ProbeResult, is_failed, is_served


class TtlRead(BaseModel):
    """One `--ttl` re-read: how long after the warm reads, and what came back."""

    offset: int
    hit: bool
    fraction: float | None = None
    offset_actual_s: float | None = None
    """When the read really went out: seconds past the warm baseline, or `None` if unknown."""

    @property
    def lateness_s(self) -> float:
        """How much later than asked for the read went out."""
        return 0.0 if self.offset_actual_s is None else max(self.offset_actual_s - self.offset, 0.0)


class RungSummary(BaseModel):
    """One rung's cold write, its warm reads and its extra requests, folded together."""

    spec: str
    rung: int
    prompt_cold: int
    cached_cold: int
    hit_rate: float
    prefix_fraction: float | None = None
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
    retries: int = 0
    skipped: bool = False
    cold_error: str | None = None


class ProbeSummary(BaseModel):
    """One spec's probe: the rungs, the pooled hit rate, and how it drifted from the rest."""

    label: str
    rungs: list[RungSummary] = Field(default_factory=list[RungSummary])
    hit_rate: float | None = None
    prefix_fraction: float | None = None
    h: float | None = None
    input_price: float | None = None
    cache_read_price: float | None = None
    price_source: str = "n/a"
    eff_per_m_prompt: float | None = None
    billed_usd: float | None = None
    providers_seen: dict[str, int] = Field(default_factory=dict)
    served: str = "-"
    """The provider names this spec's responses came from, as one string."""
    model_seen: str | None = None
    """The `model` field the responses carried; the first one when they disagree."""
    models_seen: list[str] = Field(default_factory=list[str])
    """Every distinct `model` the responses named, in first-seen order; more than one drifts."""
    tokens_delta_pct: float | None = None
    """Cold prompt tokens of the largest rung vs the reference spec's, as a percentage."""
    fingerprint_match: bool | None = None
    """The largest rung's fingerprint vs the reference spec's; `None` when either is missing."""
    ttft_ms: float | None = None
    gen_tok_s: float | None = None
    reference: bool = False
    drift: str | None = None
    """Comma-joined markers from `apply_drift`; `-` renders when it is `None`."""
    errors: int = 0
    retries: int = 0
    skipped: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def cold_ms(self) -> float | None:
        """The first rung's cold prefill: the size where a cold/warm gap is most visible."""
        first = self.rungs[0] if self.rungs else None
        return None if first is None or first.skipped else first.cold_ms

    @property
    def warm_ms(self) -> float | None:
        """The first rung's warm prefill, to be read next to `cold_ms`."""
        return self.rungs[0].warm_ms if self.rungs else None


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


def models_seen(records: Sequence[ProbeResult]) -> list[str]:
    """Every distinct `model` the served responses carried, in first-seen order.

    More than one means the endpoint routed between backends, or a gateway rewrote the
    field; either way the responses cannot all be the model that was asked for, and
    `probe_drift` is what turns that into the `model` marker.
    """
    found: list[str] = []
    for record in records:
        if is_served(record) and record.model and record.model not in found:
            found.append(record.model)
    return found


def summarize_rung(spec: str, rung: int, records: Sequence[ProbeResult]) -> RungSummary:
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
        spec=spec,
        rung=rung,
        prompt_cold=prefix,
        cached_cold=cached_of(cold),
        hit_rate=len(hits) / len(served) if served else 0.0,
        prefix_fraction=fmean(_fraction(cached_of(r), prefix) for r in hits) if hits else None,
        hits=[_hit(record, prefix) for record in warm],
        cold_ms=cold.latency_ms if cold is not None else 0.0,
        warm_ms=median(latencies) if latencies else None,
        ttft_ms=_stream_ttft(stream),
        gen_tok_s=stream.gen_tok_s if stream is not None else None,
        fingerprint=stream.fingerprint if stream is not None else None,
        ttl=[_ttl_read(record, prefix) for record in records if record.role == "ttl"],
        cache_write_cold=cold.cache_write if cold is not None else 0,
        errors=sum(1 for record in records if is_failed(record)),
        retries=sum(record.retries for record in records),
        skipped=cold is None or not is_served(cold),
        cold_error=cold.error if cold is not None else None,
    )


def summarize_probe(
    label: str,
    records: Sequence[ProbeResult],
    *,
    prices: SpecPrices | None = None,
    notes: Sequence[str] = (),
) -> ProbeSummary:
    """Pool a spec's rungs: hit rate over every served warm read, and the price it implies."""
    rungs = [summarize_rung(label, rung, _rung_records(records, rung)) for rung in _rungs(records)]
    served = [record for record in records if record.role == "warm" and is_served(record)]
    hits = [record for record in served if cached_of(record) > 0]
    hit_rate = len(hits) / len(served) if served else None
    prefix_fraction = (
        fmean(_fraction(cached_of(r), _prefix_of(records, r.rung)) for r in hits) if hits else None
    )
    h = hit_rate * (prefix_fraction or 0.0) if hit_rate is not None else None
    models = models_seen(records)
    names = sorted({record.provider for record in records if is_served(record) and record.provider})
    return ProbeSummary(
        label=label,
        rungs=rungs,
        hit_rate=hit_rate,
        prefix_fraction=prefix_fraction,
        h=h,
        input_price=prices.prices.input if prices is not None else None,
        cache_read_price=prices.prices.cache_read if prices is not None else None,
        price_source=prices.source if prices is not None else "n/a",
        eff_per_m_prompt=_effective_price(h, prices),
        billed_usd=_billed_total(records),
        providers_seen=_providers_of(records),
        served=", ".join(names) if names else "-",
        model_seen=models[0] if models else None,
        models_seen=models,
        ttft_ms=_median([rung.ttft_ms for rung in rungs if rung.ttft_ms is not None]),
        gen_tok_s=_median([rung.gen_tok_s for rung in rungs if rung.gen_tok_s is not None]),
        errors=sum(1 for record in records if is_failed(record)),
        retries=sum(record.retries for record in records),
        skipped=sum(1 for rung in rungs if rung.skipped),
        notes=[*notes, *_probe_notes(records, models)],
    )


def _stream_ttft(record: ProbeResult | None) -> float | None:
    return record.ttft_ms if record is not None and is_served(record) else None


def _ttl_read(record: ProbeResult, prefix: int) -> TtlRead:
    return TtlRead(
        offset=record.attempt,
        hit=is_served(record) and cached_of(record) > 0,
        fraction=_fraction(cached_of(record), prefix) if is_served(record) else None,
        offset_actual_s=record.offset_actual_s,
    )


def _median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def _fraction(cached: int, prefix: int) -> float:
    if prefix <= 0:
        return 0.0
    return min(cached / prefix, 1.0)


def _hit(record: ProbeResult, prefix: int) -> float | None:
    """One warm attempt's cache fraction, or `None` when the attempt failed."""
    if not is_served(record):
        return None
    return round(_fraction(cached_of(record), prefix), 2)


def _effective_price(h: float | None, prices: SpecPrices | None) -> float | None:
    """USD per 1M prompt tokens at the measured hit rate: misses at input, hits at cache read."""
    if h is None or prices is None:
        return None
    return (1 - h) * prices.prices.input + h * prices.prices.cache_read


def _billed_total(records: Sequence[ProbeResult]) -> float | None:
    """What OpenRouter says it billed, when at least one generation lookup came back."""
    billed = [amount for record in records if (amount := _billed(record)) is not None]
    return sum(billed) if billed else None


def _billed(record: ProbeResult) -> float | None:
    value = (record.generation or {}).get("total_cost")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _providers_of(records: Sequence[ProbeResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if is_served(record) and record.provider:
            counts[record.provider] = counts.get(record.provider, 0) + 1
    return counts


def _probe_notes(records: Sequence[ProbeResult], models: Sequence[str]) -> list[str]:
    """One note per distinct record note, plus the cache fallback and model-variance notes."""
    notes: list[str] = []
    for record in records:
        if record.note and record.note not in notes:
            notes.append(record.note)
    fallbacks = sum(1 for r in records if r.cached == 0 and (r.native_tokens_cached or 0) > 0)
    if fallbacks:
        notes.append(f"cached taken from the OpenRouter generation for {fallbacks} request(s)")
    if len(models) > 1:
        notes.append(f"responses named more than one model: {', '.join(models)}")
    return notes


def _rungs(records: Sequence[ProbeResult]) -> list[int]:
    return sorted({record.rung for record in records})


def _rung_records(records: Sequence[ProbeResult], rung: int) -> list[ProbeResult]:
    return [record for record in records if record.rung == rung]


def _prefix_of(records: Sequence[ProbeResult], rung: int) -> int:
    """The rung's cold prompt size, which each of its hits is measured against."""
    cold = next((r for r in records if r.rung == rung and r.role == "cold"), None)
    return cold.prompt_total if cold is not None else 0
