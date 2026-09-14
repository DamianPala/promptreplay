"""ProbeSummary: a probe run's records folded into the numbers the probe exists for.

Hit rate is hits ÷ served warm attempts — not the mean of the per-hit fractions, which
hides a skipped rung and flatters a run that never hit. The prefix fraction says how much
of the cold prefix each hit really covered, and `h` — the two multiplied — is what the
effective input price is weighted by. Both are pure functions of the records, so `report`
rebuilds them from a run directory with no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median

from pydantic import BaseModel, Field

from provibench.bench.estimate import SpecPrices
from provibench.bench.probe_models import ProbeResult, is_failed, is_served

_HIT_FULL = 0.98
"""A cached fraction at or above this renders as a full hit: Claude Code moves the marker."""
_MAX_CELL = 40
"""Longest label or provider cell rendered before it is clipped."""

_RUN_COLUMNS = (
    "spec",
    "hit %",
    "prefix %",
    "eff $/M",
    "in $/M",
    "cold ms",
    "warm ms",
    "errors",
    "providers",
)
_RUNG_COLUMNS = ("spec", "rung", "prompt", "cached cold", "hits", "cold ms", "warm ms", "errors")


class RungSummary(BaseModel):
    """One rung's cold write and its warm reads, folded into the numbers shown."""

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
    cache_write_cold: int = 0
    errors: int = 0
    retries: int = 0
    skipped: bool = False
    cold_error: str | None = None


class ProbeSummary(BaseModel):
    """One spec's probe: the rungs, the pooled hit rate, and the price it implies."""

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
    errors: int = 0
    retries: int = 0
    skipped: int = 0
    notes: list[str] = Field(default_factory=list)


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


def summarize_rung(spec: str, rung: int, records: Sequence[ProbeResult]) -> RungSummary:
    """Fold one rung's cold write and warm reads into hit rate, fractions and timings.

    A rung whose cold request was not served is `skipped`: it scores nothing — no fraction,
    an empty hit list — and carries the cold error, because a failed write is not a miss.
    """
    cold = next((record for record in records if record.role == "cold"), None)
    warm = [record for record in records if record.role == "warm"]
    served = [record for record in warm if is_served(record)]
    hits = [record for record in served if cached_of(record) > 0]
    prefix = cold.prompt_total if cold is not None else 0
    latencies = [record.latency_ms for record in served]
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
        errors=sum(1 for record in records if is_failed(record)),
        retries=sum(record.retries for record in records),
        skipped=sum(1 for rung in rungs if rung.skipped),
        notes=[*notes, *_probe_notes(records)],
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


def _probe_notes(records: Sequence[ProbeResult]) -> list[str]:
    notes: list[str] = []
    for record in records:
        if record.note and record.note not in notes:
            notes.append(record.note)
    fallbacks = sum(1 for r in records if r.cached == 0 and (r.native_tokens_cached or 0) > 0)
    if fallbacks:
        notes.append(f"cached taken from the OpenRouter generation for {fallbacks} request(s)")
    return notes


def _rungs(records: Sequence[ProbeResult]) -> list[int]:
    return sorted({record.rung for record in records})


def _rung_records(records: Sequence[ProbeResult], rung: int) -> list[ProbeResult]:
    return [record for record in records if record.rung == rung]


def _prefix_of(records: Sequence[ProbeResult], rung: int) -> int:
    """The rung's cold prompt size, which each of its hits is measured against."""
    cold = next((r for r in records if r.rung == rung and r.role == "cold"), None)
    return cold.prompt_total if cold is not None else 0


# --- rendering ----------------------------------------------------------------


def render_probe(summaries: Sequence[ProbeSummary]) -> str:
    """The probe's human output: one row per spec, then one row per rung; for a tty."""
    run_table = _table(_RUN_COLUMNS, [_run_cells(summary) for summary in summaries])
    rung_table = _table(
        _RUNG_COLUMNS, [_rung_cells(rung) for summary in summaries for rung in summary.rungs]
    )
    return "\n".join([run_table, "", rung_table, *_note_lines(summaries)])


def probe_markdown(summaries: Sequence[ProbeSummary]) -> str:
    """The same two tables as markdown, for a report that gets pasted somewhere."""
    run_table = _md_table(_RUN_COLUMNS, [_run_cells(summary) for summary in summaries])
    rung_table = _md_table(
        _RUNG_COLUMNS, [_rung_cells(rung) for summary in summaries for rung in summary.rungs]
    )
    return "\n".join([run_table, "", rung_table, "", *_note_lines(summaries)])


def _note_lines(summaries: Sequence[ProbeSummary]) -> list[str]:
    return [f"{summary.label}: {note}" for summary in summaries for note in summary.notes]


def _run_cells(summary: ProbeSummary) -> list[str]:
    return [
        _clip(summary.label, _MAX_CELL),
        _pct(summary.hit_rate),
        _pct(summary.prefix_fraction),
        _money(summary.eff_per_m_prompt, 3),
        _money(summary.input_price, 3),
        f"{_cold_ms(summary):.0f}",
        _ms(_warm_ms(summary)),
        str(summary.errors),
        _clip(_providers(summary), _MAX_CELL),
    ]


def _rung_cells(rung: RungSummary) -> list[str]:
    return [
        _clip(rung.spec, _MAX_CELL),
        str(rung.rung),
        str(rung.prompt_cold),
        str(rung.cached_cold),
        _sequence(rung.hits),
        f"{rung.cold_ms:.0f}",
        _ms(rung.warm_ms),
        str(rung.errors),
    ]


def _cold_ms(summary: ProbeSummary) -> float:
    """The median cold latency over the rungs that were served."""
    served = [rung.cold_ms for rung in summary.rungs if not rung.skipped]
    return median(served) if served else 0.0


def _warm_ms(summary: ProbeSummary) -> float | None:
    served = [rung.warm_ms for rung in summary.rungs if rung.warm_ms is not None]
    return median(served) if served else None


def _sequence(hits: Sequence[float | None]) -> str:
    """`1 1 0 1` per warm attempt: `x` failed, `1` from 0.98 up, else the fraction."""
    return " ".join(_hit_cell(hit) for hit in hits) or "-"


def _hit_cell(hit: float | None) -> str:
    if hit is None:
        return "x"
    if hit >= _HIT_FULL:
        return "1"
    return f"{hit:g}"


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}" if value is not None else "-"


def _money(value: float | None, digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _ms(value: float | None) -> str:
    return f"{value:.0f}" if value is not None else "-"


def _providers(summary: ProbeSummary) -> str:
    seen = ", ".join(f"{name}={count}" for name, count in sorted(summary.providers_seen.items()))
    return seen or "-"


def _clip(cell: str, width: int) -> str:
    return cell if len(cell) <= width else f"{cell[: width - 1]}…"


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """The header, the rule, then one line per row; a table with no rows is only a header."""
    widths = [
        max([len(columns[index]), *(len(row[index]) for row in rows)])
        for index in range(len(columns))
    ]
    lines = [_join(columns, widths), _join(["-" * width for width in widths], widths)]
    lines.extend(_join(row, widths) for row in rows)
    return "\n".join(lines)


def _md_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _join(cells: Sequence[str], widths: Sequence[int]) -> str:
    return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
