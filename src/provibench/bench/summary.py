"""RunSummary aggregation and rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from provibench.bench.openrouter import normalize_provider
from provibench.bench.pricing import CostBreakdown
from provibench.bench.replay import ReplayResult
from provibench.core.documents import Document, as_document, as_list

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


class RunSummary(BaseModel):
    label: str
    requested_providers: list[str] = Field(default_factory=list)
    turns: int
    ok: int
    errors: int
    providers_seen: dict[str, int] = Field(default_factory=dict)
    drift: int
    prompt_total: int
    cached_total: int
    cache_write_total: int
    output_total: int
    hit_ratio: float | None = None
    curve: list[float] = Field(default_factory=list[float])
    cost: CostBreakdown | None = None
    billed_total: float | None = None
    effective_per_m_prompt: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    notes: list[str] = Field(default_factory=list)


def sparkline(values: Sequence[float]) -> str:
    """Map 0..1 onto the eight-level block gradient; out-of-range values clamp."""
    if not values:
        return ""
    chars: list[str] = []
    for value in values:
        clamped = min(1.0, max(0.0, value))
        idx = min(len(_SPARK_BLOCKS) - 1, int(clamped * len(_SPARK_BLOCKS)))
        chars.append(_SPARK_BLOCKS[idx])
    return "".join(chars)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _drift(results: list[ReplayResult], requested_providers: list[str]) -> int:
    successful = [r for r in results if r.status == 200]
    if requested_providers:
        requested = {normalize_provider(p) for p in requested_providers}
        return sum(
            1 for r in successful if r.provider and normalize_provider(r.provider) not in requested
        )
    changes = 0
    previous: str | None = None
    for r in successful:
        if previous is not None and r.provider != previous:
            changes += 1
        previous = r.provider
    return changes


def _curve(results: list[ReplayResult]) -> list[float]:
    return [(r.cached / r.prompt_total) if r.prompt_total > 0 else 0.0 for r in results]


def _hit_ratio(results: list[ReplayResult]) -> float | None:
    successful = [r for r in results if r.status == 200]
    if len(successful) < 2:
        return None
    later = [r for r in successful if r.turn >= 2]
    total_prompt = sum(r.prompt_total for r in later)
    if total_prompt == 0:
        return None
    return sum(r.cached for r in later) / total_prompt


def _sum_cost(results: list[ReplayResult]) -> CostBreakdown | None:
    costs = [r.cost for r in results if r.cost is not None]
    if not costs:
        return None
    total = costs[0]
    for extra in costs[1:]:
        total = total + extra
    return total


def _sum_billed(results: list[ReplayResult]) -> float | None:
    billed = [r.billed_total for r in results if r.billed_total is not None]
    return sum(billed) if billed else None


def _effective_per_m(
    prompt_total: int, cost: CostBreakdown | None, billed_total: float | None
) -> float | None:
    if prompt_total <= 0:
        return None
    basis = billed_total if billed_total is not None else (cost.total if cost is not None else None)
    return basis / prompt_total * 1e6 if basis is not None else None


def _providers_seen(results: list[ReplayResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        if r.status == 200 and r.provider:
            counts[r.provider] = counts.get(r.provider, 0) + 1
    return counts


def _notes(results: list[ReplayResult]) -> list[str]:
    distinct: list[str] = []
    for r in results:
        if r.note and r.note not in distinct:
            distinct.append(r.note)
    return distinct


def summarize(
    label: str,
    results: list[ReplayResult],
    requested_providers: list[str],
    notes: Sequence[str] = (),
) -> RunSummary:
    ok = sum(1 for r in results if r.status == 200)
    successful_latencies = [r.latency_ms for r in results if r.status == 200]
    cost = _sum_cost(results)
    billed_total = _sum_billed(results)
    prompt_total = sum(r.prompt_total for r in results)
    return RunSummary(
        label=label,
        requested_providers=requested_providers,
        turns=len(results),
        ok=ok,
        errors=len(results) - ok,
        providers_seen=_providers_seen(results),
        drift=_drift(results, requested_providers),
        prompt_total=prompt_total,
        cached_total=sum(r.cached for r in results),
        cache_write_total=sum(r.cache_write for r in results),
        output_total=sum(r.output_tokens for r in results),
        hit_ratio=_hit_ratio(results),
        curve=_curve(results),
        cost=cost,
        billed_total=billed_total,
        effective_per_m_prompt=_effective_per_m(prompt_total, cost, billed_total),
        latency_p50_ms=_percentile(successful_latencies, 50),
        latency_p95_ms=_percentile(successful_latencies, 95),
        notes=[*notes, *_notes(results)],
    )


def cache_mode_note(warm: bool) -> str:
    """How a run treated the provider's cache, for the report's notes column."""
    return "warm" if warm else "cold (nonce)"


SUMMARY_COLUMNS = (
    "label",
    "turns ok/err",
    "drift",
    "providers seen",
    "Σprompt",
    "Σcached",
    "hit %",
    "in $",
    "cache read $",
    "cache write $",
    "out $",
    "computed $",
    "billed $",
    "eff $/M prompt",
    "p50 ms",
    "p95 ms",
)
"""The full-replay table's columns: the terminal, `report --md` and `report --html` share them."""


def summary_row(entry: Document) -> list[str]:
    """One run's cells for `SUMMARY_COLUMNS`, as the terminal prints them.

    The document carries numbers and `None`s, not strings, so this is the one place they
    become cells: the terminal, the markdown and the HTML all print these strings, so the
    three can never disagree about a rounding or a thousands separator.
    """
    providers = [d for d in map(as_document, as_list(entry.get("providers_seen")) or []) if d]
    providers_text = ", ".join(f"{p.get('provider')}={p.get('count')}" for p in providers) or "-"
    cost = as_document(entry.get("cost"))
    ratio = _number(entry.get("hit_ratio"))
    hit_pct = f"{ratio * 100:.1f}" if ratio is not None else "-"
    return [
        str(entry.get("label")),
        f"{_count(entry.get('ok'))}/{_count(entry.get('errors'))}",
        str(_count(entry.get("drift"))),
        providers_text,
        f"{_count(entry.get('prompt_total')):,}",
        f"{_count(entry.get('cached_total')):,}",
        hit_pct,
        _limit(cost.get("input") if cost else None, 4),
        _limit(cost.get("cache_read") if cost else None, 4),
        _limit(cost.get("cache_write") if cost else None, 4),
        _limit(cost.get("output") if cost else None, 4),
        _limit(cost.get("total") if cost else None, 4),
        _limit(entry.get("billed_total"), 4),
        _limit(entry.get("effective_per_m_prompt"), 3),
        _limit(entry.get("latency_p50_ms"), 0),
        _limit(entry.get("latency_p95_ms"), 0),
    ]


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


def _number(value: object) -> float | None:
    """A document's number; `None` for a missing key, a string, or a boolean."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _limit(value: object, digits: int) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.{digits}f}"


def render_markdown(entries: Sequence[Document]) -> str:
    """The summary table as markdown, with every cell `summary_row` produced.

    The input is the same `--json` shape the terminal renders from, so `report --md` and
    `report --html` on one run cannot print different numbers for it.
    """
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "---|" * len(SUMMARY_COLUMNS),
    ]
    lines.extend("| " + " | ".join(summary_row(entry)) + " |" for entry in entries)
    lines.append("")
    for entry in entries:
        curve = [float(v) for v in as_list(entry.get("curve")) or [] if isinstance(v, int | float)]
        lines.append(f"{entry.get('label')}: {sparkline(curve)}")
    return "\n".join(lines)
