"""RunSummary aggregation and rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from provibench.bench.openrouter import normalize_provider
from provibench.bench.pricing import CostBreakdown
from provibench.bench.replay import ReplayResult

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
    label: str, results: list[ReplayResult], requested_providers: list[str]
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
        notes=_notes(results),
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


_HEADERS = (
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


def _row(summary: RunSummary) -> list[str]:
    providers_items = sorted(summary.providers_seen.items())
    providers = ", ".join(f"{name}={count}" for name, count in providers_items)
    cost = summary.cost
    hit_pct = f"{summary.hit_ratio * 100:.1f}" if summary.hit_ratio is not None else "-"
    return [
        summary.label,
        f"{summary.ok}/{summary.errors}",
        str(summary.drift),
        providers or "-",
        str(summary.prompt_total),
        str(summary.cached_total),
        hit_pct,
        _fmt(cost.input if cost else None),
        _fmt(cost.cache_read if cost else None),
        _fmt(cost.cache_write if cost else None),
        _fmt(cost.output if cost else None),
        _fmt(cost.total if cost else None),
        _fmt(summary.billed_total),
        _fmt(summary.effective_per_m_prompt, 2),
        _fmt(summary.latency_p50_ms, 0),
        _fmt(summary.latency_p95_ms, 0),
    ]


def render_markdown(summaries: list[RunSummary]) -> str:
    lines = ["| " + " | ".join(_HEADERS) + " |", "|" + "---|" * len(_HEADERS)]
    lines.extend("| " + " | ".join(_row(summary)) + " |" for summary in summaries)
    lines.append("")
    lines.extend(f"{summary.label}: {sparkline(summary.curve)}" for summary in summaries)
    return "\n".join(lines)
