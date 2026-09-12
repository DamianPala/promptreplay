"""The `RunSummary` JSON shape and its shared human-readable table, used by replay and report.

`RunSummary.providers_seen` is a string-keyed map, which the O4 schema subset cannot
describe (it has no `additionalProperties`); `summary_to_document` turns it into a
`[{provider, count}]` list so the whole summary fits a fixed-property schema.

`provibench.bench.summary` pulls in httpx and pydantic (via `replay`/`openrouter`); its
symbols are imported only inside the functions that use them, or under `TYPE_CHECKING`
for annotations, so building the CLI stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    JsonSchema,
    array,
    as_document,
    as_list,
    integer,
    nullable_number,
    nullable_object,
    number,
    obj,
    string,
)
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.pricing import CostBreakdown
    from provibench.bench.summary import RunSummary

_PROVIDER_COUNT = obj({"provider": string(), "count": integer()}, required=["provider", "count"])
_COST_PROPERTIES: dict[str, JsonSchema] = {
    "input": number(),
    "cache_read": number(),
    "cache_write": number(),
    "output": number(),
    "total": number(),
    "source": string(),
}
_COST_REQUIRED = ["input", "cache_read", "cache_write", "output", "total", "source"]
SUMMARY = obj(
    {
        "label": string(),
        "requested_providers": array(string()),
        "turns": integer(),
        "ok": integer(),
        "errors": integer(),
        "providers_seen": array(_PROVIDER_COUNT),
        "drift": integer(),
        "prompt_total": integer(),
        "cached_total": integer(),
        "cache_write_total": integer(),
        "output_total": integer(),
        "hit_ratio": nullable_number(),
        "curve": array(number()),
        "cost": nullable_object(_COST_PROPERTIES, required=_COST_REQUIRED),
        "billed_total": nullable_number(),
        "effective_per_m_prompt": nullable_number(),
        "latency_p50_ms": nullable_number(),
        "latency_p95_ms": nullable_number(),
        "notes": array(string()),
    },
    required=[
        "label",
        "requested_providers",
        "turns",
        "ok",
        "errors",
        "providers_seen",
        "drift",
        "prompt_total",
        "cached_total",
        "cache_write_total",
        "output_total",
        "hit_ratio",
        "curve",
        "cost",
        "billed_total",
        "effective_per_m_prompt",
        "latency_p50_ms",
        "latency_p95_ms",
        "notes",
    ],
)

_COLUMNS = (
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


def summary_to_document(summary: RunSummary) -> Document:
    """`summary` reshaped so every field has a fixed set of JSON properties."""
    return {
        "label": summary.label,
        "requested_providers": list(summary.requested_providers),
        "turns": summary.turns,
        "ok": summary.ok,
        "errors": summary.errors,
        "providers_seen": [
            {"provider": name, "count": count}
            for name, count in sorted(summary.providers_seen.items())
        ],
        "drift": summary.drift,
        "prompt_total": summary.prompt_total,
        "cached_total": summary.cached_total,
        "cache_write_total": summary.cache_write_total,
        "output_total": summary.output_total,
        "hit_ratio": summary.hit_ratio,
        "curve": list(summary.curve),
        "cost": _cost_document(summary.cost),
        "billed_total": summary.billed_total,
        "effective_per_m_prompt": summary.effective_per_m_prompt,
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
        "notes": list(summary.notes),
    }


def _cost_document(cost: CostBreakdown | None) -> Document | None:
    if cost is None:
        return None
    return {
        "input": cost.input,
        "cache_read": cost.cache_read,
        "cache_write": cost.cache_write,
        "output": cost.output,
        "total": cost.total,
        "source": cost.source,
    }


def render_summaries(invocation: Invocation, document: Document) -> None:
    """One table of every run's totals, then one sparkline line per run."""
    from rich import box
    from rich.markup import escape
    from rich.table import Table

    from provibench.bench.summary import sparkline

    entries = [d for d in map(as_document, as_list(document.get("summaries")) or []) if d]
    table = Table(box=box.SIMPLE, header_style="bold")
    for column in _COLUMNS:
        table.add_column(column)
    for entry in entries:
        table.add_row(*(escape(escape_terminal_text(cell)) for cell in _row(entry)))
    console = invocation.stdout_console()
    console.print(table)
    for entry in entries:
        curve = [v for v in (as_list(entry.get("curve")) or []) if isinstance(v, int | float)]
        notes = ", ".join(str(note) for note in as_list(entry.get("notes")) or [])
        line = f"{entry.get('label')}  {sparkline(curve)}  {notes}"
        console.print(escape(escape_terminal_text(line)))


def _row(entry: Document) -> list[str]:
    providers = [d for d in map(as_document, as_list(entry.get("providers_seen")) or []) if d]
    providers_text = ", ".join(f"{p.get('provider')}={p.get('count')}" for p in providers) or "-"
    cost = as_document(entry.get("cost"))
    hit_ratio = entry.get("hit_ratio")
    hit_pct = f"{hit_ratio * 100:.1f}" if isinstance(hit_ratio, int | float) else "-"
    return [
        str(entry.get("label")),
        f"{_count(entry.get('ok'))}/{_count(entry.get('errors'))}",
        str(_count(entry.get("drift"))),
        providers_text,
        f"{_count(entry.get('prompt_total')):,}",
        f"{_count(entry.get('cached_total')):,}",
        hit_pct,
        _money(cost.get("input") if cost else None),
        _money(cost.get("cache_read") if cost else None),
        _money(cost.get("cache_write") if cost else None),
        _money(cost.get("output") if cost else None),
        _money(cost.get("total") if cost else None),
        _money(entry.get("billed_total")),
        _rate(entry.get("effective_per_m_prompt")),
        _fmt0(entry.get("latency_p50_ms")),
        _fmt0(entry.get("latency_p95_ms")),
    ]


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


def _money(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else "-"


def _rate(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "-"


def _fmt0(value: object) -> str:
    return f"{value:.0f}" if isinstance(value, int | float) else "-"
