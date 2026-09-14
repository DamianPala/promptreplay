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
    boolean,
    integer,
    nullable_boolean,
    nullable_number,
    nullable_object,
    nullable_string,
    number,
    obj,
    string,
)
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.pricing import CostBreakdown
    from provibench.bench.probe_summary import ProbeSummary
    from provibench.bench.summary import RunSummary


def _all(properties: dict[str, JsonSchema]) -> JsonSchema:
    """A document schema whose every property is required: these documents always carry them."""
    return obj(properties, required=list(properties))


_PROVIDER_COUNT = obj({"provider": string(), "count": integer()}, required=["provider", "count"])
_COST_PROPERTIES: dict[str, JsonSchema] = {
    "input": number(),
    "cache_read": number(),
    "cache_write": number(),
    "output": number(),
    "total": number(),
    "source": string(),
}
SUMMARY = _all(
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
        "cost": nullable_object(_COST_PROPERTIES, required=list(_COST_PROPERTIES)),
        "billed_total": nullable_number(),
        "effective_per_m_prompt": nullable_number(),
        "latency_p50_ms": nullable_number(),
        "latency_p95_ms": nullable_number(),
        "notes": array(string()),
    }
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

_TTL_READ = _all(
    {
        "offset": integer(),
        "hit": boolean(),
        "fraction": nullable_number(),
        "offset_actual_s": nullable_number(),
    }
)
_RUNG_SUMMARY = _all(
    {
        "spec": string(),
        "rung": integer(),
        "prompt_cold": integer(),
        "cached_cold": integer(),
        "hit_rate": number(),
        "prefix_fraction": nullable_number(),
        "hits": array(nullable_number()),
        "cold_ms": number(),
        "warm_ms": nullable_number(),
        "ttft_ms": nullable_number(),
        "gen_tok_s": nullable_number(),
        "fingerprint": nullable_string(),
        "ttl": array(_TTL_READ),
        "cache_write_cold": integer(),
        "errors": integer(),
        "retries": integer(),
        "skipped": boolean(),
        "cold_error": nullable_string(),
    }
)
PROBE_SUMMARY = _all(
    {
        "label": string(),
        "rungs": array(_RUNG_SUMMARY),
        "hit_rate": nullable_number(),
        "prefix_fraction": nullable_number(),
        "h": nullable_number(),
        "input_price": nullable_number(),
        "cache_read_price": nullable_number(),
        "price_source": string(),
        "eff_per_m_prompt": nullable_number(),
        "billed_usd": nullable_number(),
        "providers_seen": array(_PROVIDER_COUNT),
        "served": string(),
        "model_seen": nullable_string(),
        "models_seen": array(string()),
        "tokens_delta_pct": nullable_number(),
        "fingerprint_match": nullable_boolean(),
        "ttft_ms": nullable_number(),
        "gen_tok_s": nullable_number(),
        "reference": boolean(),
        "drift": nullable_string(),
        "errors": integer(),
        "retries": integer(),
        "skipped": integer(),
        "notes": array(string()),
    }
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


def probe_summary_to_document(summary: ProbeSummary) -> Document:
    """`summary` reshaped so `providers_seen` has a fixed set of JSON properties."""
    return {
        "label": summary.label,
        "rungs": [rung.model_dump() for rung in summary.rungs],
        "hit_rate": summary.hit_rate,
        "prefix_fraction": summary.prefix_fraction,
        "h": summary.h,
        "input_price": summary.input_price,
        "cache_read_price": summary.cache_read_price,
        "price_source": summary.price_source,
        "eff_per_m_prompt": summary.eff_per_m_prompt,
        "billed_usd": summary.billed_usd,
        "providers_seen": [
            {"provider": name, "count": count}
            for name, count in sorted(summary.providers_seen.items())
        ],
        "served": summary.served,
        "model_seen": summary.model_seen,
        "models_seen": list(summary.models_seen),
        "tokens_delta_pct": summary.tokens_delta_pct,
        "fingerprint_match": summary.fingerprint_match,
        "ttft_ms": summary.ttft_ms,
        "gen_tok_s": summary.gen_tok_s,
        "reference": summary.reference,
        "drift": summary.drift,
        "errors": summary.errors,
        "retries": summary.retries,
        "skipped": summary.skipped,
        "notes": list(summary.notes),
    }


def render_probe_run(invocation: Invocation, document: Document) -> None:
    """Human rendering of a `probe` result: the spec and rung tables."""
    _render_probe(invocation, document.get("summaries"))


def probe_tables_text(document: Document) -> str:
    """The probe run's two tables as text, for a caller that writes them itself."""
    from provibench.bench.probe_tables import render_probe

    entries = [d for d in map(as_document, as_list(document.get("summaries")) or []) if d]
    return render_probe([_probe_summary(entry) for entry in entries])


def render_probe_report(invocation: Invocation, document: Document) -> None:
    """Human rendering of `report` for a probe run: the same two tables."""
    _render_probe(invocation, document.get("probe_summaries"))


def render_report_document(invocation: Invocation, document: Document) -> None:
    """`report`'s human rendering: the probe tables for a probe run, else the replay table."""
    if document.get("protocol") == "probe":
        render_probe_report(invocation, document)
        return
    render_summaries(invocation, document)


def _render_probe(invocation: Invocation, value: object) -> None:
    """The probe tables are plain text, so they render the same at any terminal width."""
    from provibench.bench.probe_tables import render_probe

    entries = [d for d in map(as_document, as_list(value) or []) if d]
    summaries = [_probe_summary(entry) for entry in entries]
    stdout = invocation.streams.stdout
    escaped = [escape_terminal_text(line) for line in render_probe(summaries).splitlines()]
    stdout.write("\n".join(escaped) + "\n")
    stdout.flush()


def _probe_summary(entry: Document) -> ProbeSummary:
    """The document shape back into a `ProbeSummary`: `providers_seen` is a list there."""
    from provibench.bench.probe_summary import ProbeSummary

    providers = [d for d in map(as_document, as_list(entry.get("providers_seen")) or []) if d]
    seen = {str(p.get("provider")): _count(p.get("count")) for p in providers}
    return ProbeSummary.model_validate({**entry, "providers_seen": seen})


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
