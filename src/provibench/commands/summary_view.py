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
    nullable_integer,
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

_SWEEP_DROP = _all({"tag": string(), "spec": string(), "reason": string(), "checked": boolean()})
_SWEEP_RANKED = _all(
    {
        "tag": string(),
        "price_input": number(),
        "uptime_1d": nullable_number(),
        "status": nullable_string(),
        "latency_p50_ms": nullable_number(),
        "throughput_p50_tok_s": nullable_number(),
    }
)
SWEEP_BLOCK_PROPERTIES: dict[str, JsonSchema] = {
    "model": string(),
    "target": string(),
    "included": array(string()),
    "excluded": array(string()),
    "sort": string(),
    "top": nullable_integer(),
    "zdr": boolean(),
    "check": boolean(),
    "dropped": array(_SWEEP_DROP),
    "ranking": array(_SWEEP_RANKED),
}
"""The `sweep` block's fields: `sweep` publishes them, `report` re-reads them as nullable."""
SWEEP_BLOCK = obj(SWEEP_BLOCK_PROPERTIES, required=list(SWEEP_BLOCK_PROPERTIES))


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
    _render_probe(invocation, document, key="summaries")


def probe_report_text(document: Document, *, key: str = "summaries") -> str:
    """The probe run's tables and its selection as text, for a caller that writes its own.

    A sweep's run says how its endpoints were chosen under the same tables, so a caller that
    prints this block itself shows the selection line and the `not probed` notes as well —
    which is the failure path, where the reader most wants to know which candidates were
    removed before the requests that failed.
    """
    from provibench.bench.probe_tables import render_probe

    entries = [d for d in map(as_document, as_list(document.get(key)) or []) if d]
    lines = [
        *render_probe([_probe_summary(entry) for entry in entries]).splitlines(),
        *sweep_lines(document.get("sweep")),
    ]
    return "\n".join(lines)


def render_probe_report(invocation: Invocation, document: Document) -> None:
    """Human rendering of `report` for a probe run: the same two tables."""
    _render_probe(invocation, document, key="probe_summaries")


def render_report_document(invocation: Invocation, document: Document) -> None:
    """`report`'s human rendering: the probe tables for a probe run, else the replay table."""
    if document.get("protocol") == "probe":
        render_probe_report(invocation, document)
    else:
        render_summaries(invocation, document)
    _render_html_path(invocation, document)


def _render_html_path(invocation: Invocation, document: Document) -> None:
    """The HTML file's path, last: it is the one thing a caller with --html came for."""
    from rich.markup import escape

    path = document.get("html")
    if not isinstance(path, str):
        return
    console = invocation.stdout_console()
    console.print(escape(f"HTML report: {escape_terminal_text(path)}"))


def _render_probe(invocation: Invocation, document: Document, *, key: str) -> None:
    """The probe tables are plain text, so they render the same at any terminal width."""
    lines = probe_report_text(document, key=key).splitlines()
    stdout = invocation.streams.stdout
    escaped = [escape_terminal_text(line) for line in lines]
    stdout.write("\n".join(escaped) + "\n")
    stdout.flush()


def message_lines(invocation: Invocation, text: str) -> None:
    """A block of text as one message per line.

    `Invocation.message` escapes control characters, a newline included, so a whole table
    handed to it would land as a single line the reader cannot line up.
    """
    for line in text.splitlines():
        invocation.message(line)


def sweep_lines(block: object) -> list[str]:
    """A recorded selection as the lines that follow the tables; none without a sweep."""
    from provibench.bench.selection import SweepInfo, not_probed_lines, selection_line

    document = as_document(block)
    if document is None:
        return []
    sweep = SweepInfo.model_validate(document)
    return [selection_line(sweep), *not_probed_lines(sweep)]


def _probe_summary(entry: Document) -> ProbeSummary:
    """The document shape back into a `ProbeSummary`: `providers_seen` is a list there."""
    from provibench.bench.probe_summary import summary_from_document

    return summary_from_document(entry)


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

    from provibench.bench.summary import SUMMARY_COLUMNS, sparkline, summary_row

    entries = [d for d in map(as_document, as_list(document.get("summaries")) or []) if d]
    table = Table(box=box.SIMPLE, header_style="bold")
    for column in SUMMARY_COLUMNS:
        table.add_column(column)
    for entry in entries:
        table.add_row(*(escape(escape_terminal_text(cell)) for cell in summary_row(entry)))
    console = invocation.stdout_console()
    console.print(table)
    for entry in entries:
        curve = [v for v in (as_list(entry.get("curve")) or []) if isinstance(v, int | float)]
        notes = ", ".join(str(note) for note in as_list(entry.get("notes")) or [])
        line = f"{entry.get('label')}  {sparkline(curve)}  {notes}"
        console.print(escape(escape_terminal_text(line)))
