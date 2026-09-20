"""The shared human-readable rendering for `RunSummary` and `ProbeSummary` documents.

Used by `replay`, `probe`, `sweep` and `report`; the JSON shapes those summaries reshape
into live in `summary_schema`, split out purely for this module's line budget.

`provibench.bench.summary` pulls in httpx and pydantic (via `replay`/`openrouter`); its
symbols are imported only inside the functions that use them, or under `TYPE_CHECKING`
for annotations, so building the CLI stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from provibench.commands.probe_spend import session_footer_lines, spend_lines
from provibench.commands.summary_schema import (
    PROBE_SUMMARY,
    SUMMARY,
    SWEEP_BLOCK,
    SWEEP_BLOCK_PROPERTIES,
    probe_summary_to_document,
    summary_to_document,
)
from provibench.core.context import Invocation
from provibench.core.documents import Document, as_document, as_list
from provibench.core.output import Format
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.probe_summary import ProbeSummary

__all__ = [
    "PROBE_SUMMARY",
    "SUMMARY",
    "SWEEP_BLOCK",
    "SWEEP_BLOCK_PROPERTIES",
    "message_lines",
    "probe_report_text",
    "probe_summary_to_document",
    "render_probe_report",
    "render_probe_run",
    "render_report_document",
    "render_summaries",
    "report_markdown",
    "summary_to_document",
    "sweep_lines",
]


def render_probe_run(invocation: Invocation, document: Document) -> None:
    """Human rendering of a `probe` result: the endpoint and rung tables, or the dry-run notice."""
    if _render_dry_run_notice(invocation, document):
        return
    _render_probe(invocation, document, key="summaries")


def probe_report_text(document: Document, *, key: str = "summaries") -> str:
    """The probe run's tables and its selection as text, for a caller that writes its own.

    A sweep's run says how its endpoints were chosen under the same tables, so a caller that
    prints this block itself shows the selection line and the `not probed` notes as well —
    which is the failure path, where the reader most wants to know which candidates were
    removed before the requests that failed.
    """
    from provibench.bench.probe_tables import probe_labels, render_probe

    entries = [d for d in map(as_document, as_list(document.get(key)) or []) if d]
    summaries = [_probe_summary(entry) for entry in entries]
    labels = probe_labels(summaries)
    trace_tokens = document.get("trace_prompt_tokens")
    closing = [
        *sweep_lines(document.get("sweep")),
        *spend_lines(document),
        *session_footer_lines(summaries, labels, trace_tokens),
    ]
    lines = render_probe(summaries).splitlines()
    if closing:
        lines = [*lines, "", *closing]
    return "\n".join(lines)


def render_probe_report(invocation: Invocation, document: Document) -> None:
    """Human rendering of `report` for a probe run: the same two tables."""
    _render_probe(invocation, document, key="summaries")


def render_report_document(invocation: Invocation, document: Document) -> None:
    """`report`'s human rendering: the probe tables for a probe run, else the replay table."""
    if invocation.format is Format.MARKDOWN:
        _write_text(invocation, report_markdown(document))
        return
    if invocation.format is Format.HTML:
        from provibench.bench.html_report import render_html

        _write_text(invocation, render_html(document))
        return
    if document.get("protocol") == "probe":
        render_probe_report(invocation, document)
    else:
        render_summaries(invocation, document)


def report_markdown(document: Document) -> str:
    """Render the report's Markdown form from the same document as every other format."""
    if document.get("protocol") == "probe":
        from provibench.bench.probe_tables import probe_labels, probe_markdown

        entries = [
            entry for entry in map(as_document, as_list(document.get("summaries")) or []) if entry
        ]
        summaries = [_probe_summary(entry) for entry in entries]
        labels = probe_labels(summaries)
        trace_tokens = document.get("trace_prompt_tokens")
        closing = [*spend_lines(document), *session_footer_lines(summaries, labels, trace_tokens)]
        lines = probe_markdown(summaries).splitlines()
        if closing:
            lines = [*lines, "", *closing]
        return "\n".join(lines) + "\n"
    from provibench.bench.summary import render_markdown

    entries = [
        entry for entry in map(as_document, as_list(document.get("summaries")) or []) if entry
    ]
    return render_markdown(entries) + "\n"


def _write_text(invocation: Invocation, text: str) -> None:
    invocation.streams.stdout.write(text)
    invocation.streams.stdout.flush()


def _render_probe(invocation: Invocation, document: Document, *, key: str) -> None:
    """The probe tables are plain text, so they render the same at any terminal width."""
    lines = probe_report_text(document, key=key).splitlines()
    stdout = invocation.streams.stdout
    escaped = [escape_terminal_text(line) for line in lines]
    stdout.write("\n".join(escaped) + "\n")
    stdout.flush()


def _render_dry_run_notice(invocation: Invocation, document: Document) -> bool:
    """Print the dry-run notice and report `True`, or do nothing and report `False`.

    `estimate` only appears in a `--dry-run` result (`commands.dry_run.build_document`); the
    tables the estimate itself takes are already on stderr, printed before the confirmation
    a dry run never reaches, so the result's own rendering is this one line.
    """
    if "estimate" not in document:
        return False
    stdout = invocation.streams.stdout
    stdout.write("dry run: nothing sent\n")
    stdout.flush()
    return True


def message_lines(invocation: Invocation, text: str) -> None:
    """A block of text as one message per line.

    `Invocation.message` escapes control characters, a newline included, so a whole table
    handed to it would land as a single line the reader cannot line up.
    """
    for line in text.splitlines():
        invocation.message(line)


def sweep_lines(block: object) -> list[str]:
    """A recorded selection as the lines that follow the tables; none without a sweep."""
    from provibench.bench.selection import SweepInfo
    from provibench.bench.selection_text import not_probed_lines, selection_line

    document = as_document(block)
    if document is None:
        return []
    sweep = SweepInfo.model_validate(document)
    return [selection_line(sweep), *not_probed_lines(sweep)]


def _probe_summary(entry: Document) -> ProbeSummary:
    """The document shape back into a `ProbeSummary`: `providers_seen` is a list there."""
    from provibench.bench.probe_summary import summary_from_document

    return summary_from_document(entry)


def render_summaries(invocation: Invocation, document: Document) -> None:
    """One section: an `endpoints:` caption, the totals table, then one sparkline footer per
    endpoint, or the dry-run notice.

    The caption names endpoints, not runs: one `report` shows one run, and each row is one
    endpoint's replay of it -- the same thing every other table in the tool captions.
    """
    if _render_dry_run_notice(invocation, document):
        return
    from provibench.bench.labels import text_table
    from provibench.bench.summary import SUMMARY_COLUMNS, sparkline, summary_row

    entries = [d for d in map(as_document, as_list(document.get("summaries")) or []) if d]
    lines = ["endpoints:", *text_table(SUMMARY_COLUMNS, [summary_row(entry) for entry in entries])]
    for entry in entries:
        curve = [v for v in (as_list(entry.get("curve")) or []) if isinstance(v, int | float)]
        notes = ", ".join(str(note) for note in as_list(entry.get("notes")) or [])
        lines.append(f"{entry.get('label')}  {sparkline(curve)}  {notes}")
    stdout = invocation.streams.stdout
    stdout.write("\n".join(escape_terminal_text(line) for line in lines) + "\n")
    stdout.flush()
