"""ProbeSummary: a probe run's records folded into the numbers the probe exists for.

Hit rate is hits ÷ served warm attempts, not the mean of the per-hit fractions, which hides
a skipped rung and flatters a run that never hit. The cached fraction is how much of the
cold prefix each hit covered, and `h` — the two multiplied — weights the effective input
price. Both are pure functions of the records, so `report` rebuilds them with no network.

The pooled `hit_rate` stands in for a long session's steady state: by read 2 the prefix has
been seen twice (some caches admit it only then) and more than one replica is warm. Read 1
alone is the cold-start bound, one write then one read, which is what
`first_hit_rate`/`first_cached_fraction`/`first_h` report next to it. `eff_per_m_prompt`
is priced from the pooled `h`; the gap to `first_h` is what the first minutes cost.

The other question the probe exists for — which endpoint is fast, and which is really
serving the model — is answered here and by the comparisons in `probe_drift`.

A rung's own numbers (`RungSummary`, `TtlRead`, `cached_of`, `summarize_rung`) live in
`bench.probe_rung_summary`, split out purely for this module's line budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean, median

from pydantic import BaseModel, Field

from promptreplay.bench.estimate import SpecPrices
from promptreplay.bench.pricing import effective_price
from promptreplay.bench.probe_models import ProbeResult, is_failed, is_served
from promptreplay.bench.probe_notes import probe_notes
from promptreplay.bench.probe_pricing import choose_prices
from promptreplay.bench.probe_reads import (
    first_read_stats,
    output_token_stats,
    output_tokens_note,
    rate_limited_count,
)
from promptreplay.bench.probe_rung_summary import RungSummary, cached_of, summarize_rung
from promptreplay.bench.spend import output_cost, spec_spend
from promptreplay.bench.targets import Prices
from promptreplay.core.documents import Document, as_document, as_list


class ProbeSummary(BaseModel):
    """One spec's probe: the rungs, the pooled hit rate, and how it drifted from the rest."""

    label: str
    rungs: list[RungSummary] = Field(default_factory=list[RungSummary])
    hit_rate: float | None = None
    cached_fraction: float | None = None
    h: float | None = None
    first_hit_rate: float | None = None
    first_cached_fraction: float | None = None
    first_h: float | None = None
    """First-read analogues of `hit_rate`/`cached_fraction`/`h` (see the module docstring)."""
    input_price: float | None = None
    cache_read_price: float | None = None
    price_source: str = "n/a"
    priced_as: str = "n/a"
    """How the two prices above were chosen: `pinned`, `served: <provider(s)> (listed)` or
    `(rates fitted from this run's billed records)` (`...weighted` for several providers),
    or `worst case: <provider>` when none could be repriced this way."""
    listed_input: float | None = None
    listed_cache_read: float | None = None
    listed_source: str = "n/a"
    """What `history`/`compare` read as this spec's listed price: `input_price` unchanged for
    a pinned spec or a listing reprice, else the worst case, so a noisier fitted reprice
    never reads there as a genuine price change between two runs (`bench.probe_pricing`)."""
    eff_per_m_prompt: float | None = None
    billed_usd: float | None = None
    spend_usd: float | None = None
    """What this spec cost: `billed_usd` when OpenRouter reported one, else the usage priced
    at the two prices above -- how a native spec, which never gets a `billed_usd`, gets one."""
    session_prompt_usd: float | None = None
    """`eff_per_m_prompt` times the trace's total prompt tokens: a session-shaped bill."""
    output_tokens: int = 0
    """Output tokens over every served read; a surplus over the read count means a provider
    ignored `max_tokens: 1`."""
    output_usd: float | None = None
    """Those output tokens billed at the listed output price; `None` without one. This is
    on top of `spend_usd`, not part of the worst-case estimate, which prices prompt tokens
    only -- `output_tokens_note` is where it turns into the reader-facing dollar clause."""
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
    rate_limited: int = 0
    """Of `errors`, how many were an HTTP 429; see `notes` for the count when it is non-zero."""
    retries: int = 0
    skipped: int = 0
    notes: list[str] = Field(default_factory=list)
    # OpenRouter's reported average for this provider that day: share, whether it pools more
    # than one endpoint, the same two prices/bill at that share, and the bill against it
    # (negative `vs_or_avg_pct`: cached better) -- see `bench.reported_average`.
    or_avg_share_pct: float | None = None
    or_avg_pooled: bool = False
    or_avg_eff_per_m_prompt: float | None = None
    or_avg_session_prompt_usd: float | None = None
    vs_or_avg_pct: float | None = None

    @property
    def cold_ms(self) -> float | None:
        """The first rung's cold prefill: the size where a cold/warm gap is most visible."""
        first = self.rungs[0] if self.rungs else None
        return None if first is None or first.skipped else first.cold_ms

    @property
    def warm_ms(self) -> float | None:
        """The first rung's warm prefill, to be read next to `cold_ms`."""
        return self.rungs[0].warm_ms if self.rungs else None


def summary_from_document(entry: Document) -> ProbeSummary:
    """A `--json` probe summary back into a `ProbeSummary`.

    The document carries `providers_seen` as a list of `{provider, count}` pairs — the O4
    schema subset has no `additionalProperties` — so it is folded back into the mapping
    the model holds. Both the terminal report and the HTML report read a document and go
    through here, so they summarise the same numbers.
    """
    providers = [d for d in map(as_document, as_list(entry.get("providers_seen")) or []) if d]
    seen = {str(p.get("provider")): _count(p.get("count")) for p in providers}
    return ProbeSummary.model_validate({**entry, "providers_seen": seen})


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


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


def summarize_probe(
    label: str,
    records: Sequence[ProbeResult],
    *,
    prices: SpecPrices | None = None,
    notes: Sequence[str] = (),
    unpinned_gateway: bool = False,
    trace_prompt_tokens: int | None = None,
    listing: Mapping[str, Prices] | None = None,
    warm: bool = False,
) -> ProbeSummary:
    """Pool a spec's rungs: hit rate over every served warm read, and the price it implies.

    `unpinned_gateway` is whether this spec is an OpenRouter spec with no pinned provider --
    the only case `priced_as` reprices from what actually served it rather than keeping
    `prices` as given (see `bench.probe_pricing.choose_prices`). `listing` is the run's own
    endpoint listing, passed through to `choose_prices`, which prefers it over fitting a
    served provider's rate from its billed records. `trace_prompt_tokens`, when known, turns
    the effective price into `session_prompt_usd`, the number a session like the trace bills.
    `warm` silences the cached-cold caveat: under `--warm` a cold write hitting the cache is
    the thing the run asked to measure, not a surprise.
    """
    rungs = [summarize_rung(label, rung, _rung_records(records, rung)) for rung in _rungs(records)]
    served = [record for record in records if record.role == "warm" and is_served(record)]
    hits = [record for record in served if cached_of(record) > 0]
    hit_rate = len(hits) / len(served) if served else None
    cached_fraction = (
        fmean(_fraction(cached_of(r), _prefix_of(records, r.rung)) for r in hits) if hits else None
    )
    h = hit_rate * (cached_fraction or 0.0) if hit_rate is not None else None
    first_hit_rate, first_cached_fraction, first_h = first_read_stats(records)
    models = models_seen(records)
    names = sorted({record.provider for record in records if is_served(record) and record.provider})
    priced, priced_as, listed = choose_prices(
        prices, records, unpinned_gateway=unpinned_gateway, listing=listing
    )
    # Priced from the pooled `h`, the steady-state rate; `first_h` is the cold-start bound
    # and stays visible next to it (see the module docstring).
    eff_per_m_prompt = effective_price(h, priced.prices if priced is not None else None)
    billed_usd = _billed_total(records)
    output_tokens, reads = output_token_stats(records)
    output_usd = output_cost(records, priced)
    rate_limited = rate_limited_count(records)
    all_notes = [*notes, *probe_notes(records, models, warm=warm)]
    all_notes.extend(
        note
        for note in (
            f"{rate_limited} x 429 rate limit" if rate_limited else None,
            output_tokens_note(output_tokens, reads, output_usd=output_usd),
            # The table's `in $/M` cell cannot say where its number came from, and for an
            # unpinned spec that is not the listed price the estimate showed; the note is
            # where a reader of the rendered report is told.
            f"priced as {priced_as}" if priced_as != "pinned" else None,
        )
        if note is not None
    )
    return ProbeSummary(
        label=label,
        rungs=rungs,
        hit_rate=hit_rate,
        cached_fraction=cached_fraction,
        h=h,
        first_hit_rate=first_hit_rate,
        first_cached_fraction=first_cached_fraction,
        first_h=first_h,
        input_price=priced.prices.input if priced is not None else None,
        cache_read_price=priced.prices.cache_read if priced is not None else None,
        price_source=priced.source if priced is not None else "n/a",
        priced_as=priced_as,
        listed_input=listed.prices.input if listed is not None else None,
        listed_cache_read=listed.prices.cache_read if listed is not None else None,
        listed_source=listed.source if listed is not None else "n/a",
        eff_per_m_prompt=eff_per_m_prompt,
        billed_usd=billed_usd,
        spend_usd=spec_spend(records, billed_usd=billed_usd, prices=priced),
        session_prompt_usd=(
            eff_per_m_prompt * trace_prompt_tokens / 1e6
            if eff_per_m_prompt is not None and trace_prompt_tokens is not None
            else None
        ),
        output_tokens=output_tokens,
        output_usd=output_usd,
        providers_seen=_providers_of(records),
        served=", ".join(names) if names else "-",
        model_seen=models[0] if models else None,
        models_seen=models,
        ttft_ms=_median([rung.ttft_ms for rung in rungs if rung.ttft_ms is not None]),
        gen_tok_s=_median([rung.gen_tok_s for rung in rungs if rung.gen_tok_s is not None]),
        errors=sum(1 for record in records if is_failed(record)),
        rate_limited=rate_limited,
        retries=sum(record.retries for record in records),
        skipped=sum(1 for rung in rungs if rung.skipped),
        notes=all_notes,
    )


def _median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def _fraction(cached: int, prefix: int) -> float:
    if prefix <= 0:
        return 0.0
    return min(cached / prefix, 1.0)


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


def _rungs(records: Sequence[ProbeResult]) -> list[int]:
    return sorted({record.rung for record in records})


def _rung_records(records: Sequence[ProbeResult], rung: int) -> list[ProbeResult]:
    return [record for record in records if record.rung == rung]


def _prefix_of(records: Sequence[ProbeResult], rung: int) -> int:
    """The rung's cold prompt size, which each of its hits is measured against."""
    cold = next((r for r in records if r.rung == rung and r.role == "cold"), None)
    return cold.prompt_total if cold is not None else 0
