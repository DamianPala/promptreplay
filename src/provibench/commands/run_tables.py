"""The `history` and `compare` tables, written as plain text from their `--json` documents.

The documents are the source of truth (`commands.run_view` builds them): the terminal
renders the same strings a script parses, so the two cannot disagree about a rounding. The
layout goes through `bench.labels`, like the probe tables, which is what shortens a column
of near-identical spec labels into a caption plus `@tag` tails — the same rule, and the
same `text_table`, as everywhere else in the tool.

The numbers stay what the runs measured: a hit rate is a fraction in the document, and a
table prints it as a per cent because that is how two of them are compared at a glance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from provibench.core.context import Invocation
from provibench.core.documents import Document, as_document, as_list
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from collections.abc import Sequence

_LABEL_CAP = 40
"""Widest the spec column is planned to be; past this a row's name is elided."""
_PCT_DIGITS = 1
_USD_DIGITS = 3
_MS_DIGITS = 0
_TOK_S_DIGITS = 1
_RUNS = "run(s)"
"""What the sparkline block counts: one run is as ordinary here as thirty."""
_ARROW = " → "
_HISTORY_COLUMNS = (
    "date",
    "trace",
    "protocol",
    "spec",
    "hit %",
    "eff $/M",
    "TTFT ms",
    "tok/s",
    "errors",
    "in $/M",
)
_COMPARE_COLUMNS = ("spec", "hit %", "eff $/M", "TTFT ms", "tok/s")
_LISTED_COLUMNS = ("spec", "listed $/M in", "listed cache read $/M")
_NO_SNAPSHOT = "listed prices: no spec was priced in both runs"


def history_text(document: Document) -> str:
    """The history table, then one sparkline line per spec, trace and protocol."""
    rows = _rows(document, "rows")
    series = _rows(document, "series")
    caption, labels = _labels([str(entry.get("spec")) for entry in series])
    traces = {str(entry.get("trace")) for entry in series}
    protocols = {str(entry.get("protocol")) for entry in series}
    lines = [
        *([caption] if caption is not None else []),
        *_table(_HISTORY_COLUMNS, [_history_cells(row, labels) for row in rows]),
    ]
    if series:
        lines.append("")
        named = [
            _series_line(
                entry, labels, show_trace=len(traces) > 1, show_protocol=len(protocols) > 1
            )
            for entry in series
        ]
        # the names are padded to one width so every sparkline starts in the same column
        width = max(len(name) for name, _ in named)
        lines.extend(f"{name.ljust(width)}  {bars}" for name, bars in named)
    return "\n".join(lines)


def compare_text(document: Document) -> str:
    """What the two runs are, the delta per metric, then the listed prices and the unpaired."""
    rows = _rows(document, "rows")
    caption, labels = _labels([str(row.get("spec")) for row in rows])
    lines = [
        f"A: {_run_line(document.get('run_a'))}",
        f"B: {_run_line(document.get('run_b'))}",
        "",
        *([caption] if caption is not None else []),
        *_table(_COMPARE_COLUMNS, [_compare_cells(row, labels) for row in rows]),
    ]
    listed = _rows(document, "listed")
    if listed:
        lines.extend(["", *_table(_LISTED_COLUMNS, [_listed_cells(row, labels) for row in listed])])
    elif rows:
        # The absence is said out loud: the listed price is the one number a comparison
        # cannot recompute, and a reader is owed the reason it is missing.
        lines.extend(["", _NO_SNAPSHOT])
    for key, label in (("only_in_a", "only in A"), ("only_in_b", "only in B")):
        specs = [str(spec) for spec in as_list(document.get(key)) or []]
        if specs:
            lines.append(f"{label}: {', '.join(specs)}")
    return "\n".join(lines)


def render_history(invocation: Invocation, document: Document) -> None:
    """`history`'s human rendering: the table and the sparklines, written as plain text."""
    _write_lines(invocation, history_text(document))


def render_compare(invocation: Invocation, document: Document) -> None:
    """`compare`'s human rendering: the two run names, the deltas and the unpaired specs."""
    _write_lines(invocation, compare_text(document))


def _write_lines(invocation: Invocation, text: str) -> None:
    """A block of plain text on stdout, escaped like every other table the tool prints."""
    stdout = invocation.streams.stdout
    stdout.write("\n".join(escape_terminal_text(line) for line in text.splitlines()) + "\n")
    stdout.flush()


def _table(columns: Sequence[str], rows: list[list[str]]) -> list[str]:
    from provibench.bench.labels import text_table

    return text_table(columns, rows)


def _rows(document: Document, key: str) -> list[Document]:
    return [d for d in map(as_document, as_list(document.get(key)) or []) if d]


def _labels(specs: Sequence[str]) -> tuple[str | None, dict[str, str]]:
    """The caption and one label per spec, shortened by the rule every other table uses."""
    from provibench.bench.labels import column_labels, rendered

    caption, column = column_labels(specs, label_width=min(_LABEL_CAP, rendered(specs)))
    return caption, dict(zip(specs, column, strict=True))


def _history_cells(row: Document, labels: dict[str, str]) -> list[str]:
    return [
        _stamp(row.get("created")),
        str(row.get("trace")),
        str(row.get("protocol")),
        _label(row, labels),
        _number(row.get("hit_rate"), _PCT_DIGITS, scale=100.0),
        _number(row.get("eff_per_m_prompt"), _USD_DIGITS),
        _number(row.get("ttft_ms"), _MS_DIGITS),
        _number(row.get("gen_tok_s"), _TOK_S_DIGITS),
        str(_count(row.get("errors"))),
        _number(row.get("listed_input"), _USD_DIGITS),
    ]


def _compare_cells(row: Document, labels: dict[str, str]) -> list[str]:
    return [
        _label(row, labels),
        _metric(row.get("hit_rate"), _PCT_DIGITS, percent=True),
        _metric(row.get("eff_per_m_prompt"), _USD_DIGITS),
        _metric(row.get("ttft_ms"), _MS_DIGITS),
        _metric(row.get("gen_tok_s"), _TOK_S_DIGITS),
    ]


def _listed_cells(row: Document, labels: dict[str, str]) -> list[str]:
    return [
        _label(row, labels),
        _metric(row.get("price_in"), _USD_DIGITS),
        _metric(row.get("price_cache_read"), _USD_DIGITS),
    ]


def _series_line(
    entry: Document,
    labels: dict[str, str],
    *,
    show_trace: bool,
    show_protocol: bool,
) -> tuple[str, str]:
    """One spec's name, and its sparkline with its run count, labelled as its rows are.

    A run that measured no hit rate has `null` in the series. It gets a placeholder so the
    bars remain aligned with the run count and the time axis. The name comes back apart
    from the bars because the caller pads the names of the whole block to one width.
    """
    from provibench.bench.summary import sparkline

    spec = str(entry.get("spec"))
    rates = as_list(entry.get("hit_rates")) or []
    bars = [sparkline([float(value)]) if isinstance(value, int | float) else "·" for value in rates]
    name = labels.get(spec, spec)
    if show_trace:
        name += f"  trace={entry.get('trace')}"
    if show_protocol:
        name += f"  protocol={entry.get('protocol')}"
    return name, f"{''.join(bars) or '-'}  {_count(entry.get('runs'))} {_RUNS}"


def _label(row: Document, labels: dict[str, str]) -> str:
    spec = str(row.get("spec"))
    return labels.get(spec, spec)


def _metric(value: object, digits: int, *, percent: bool = False) -> str:
    """`A → B (+Δ)` for one metric; `-` stands in wherever a run measured nothing."""
    pair = as_document(value) or {}
    scale = 100.0 if percent else 1.0
    left = _number(pair.get("a"), digits, scale=scale)
    right = _number(pair.get("b"), digits, scale=scale)
    delta = _number(pair.get("delta"), digits, scale=scale, signed=True)
    return f"{left}{_ARROW}{right} ({delta})"


def _number(value: object, digits: int, *, scale: float = 1.0, signed: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    scaled = float(value) * scale
    return f"{scaled:+.{digits}f}" if signed else f"{scaled:.{digits}f}"


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


def _stamp(created: object) -> str:
    """A run's stamp as a date and a minute: `2027-01-14 08:00`, unreadable stamps as they are."""
    from provibench.bench.run_history import created_at

    text = str(created)
    at = created_at(text)
    if at is None:
        return text
    return datetime.fromtimestamp(at, UTC).strftime("%Y-%m-%d %H:%M")


def _run_line(value: object) -> str:
    """One of the two runs as a line: the directory, the stamp and the protocol."""
    run = as_document(value) or {}
    return f"{run.get('run_dir')}  {_stamp(run.get('created'))}  {run.get('protocol')}"
