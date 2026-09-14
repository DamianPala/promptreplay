"""The probe's two tables: one row per spec, one row per rung, as text or markdown.

Kept apart from `probe_summary` (which folds the records into the numbers) because the
rendering has its own problems: the spec column has to stay readable when every spec shares
the `target:model` prefix without ever clipping two specs into the same string, a column
that only exists when `--ttl` ran must not leave a hole when it did not, and the whole
thing has to fit 120 columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from provibench.bench.labels import column_labels, elide, named, rendered, text_table
from provibench.bench.probe_summary import ProbeSummary, RungSummary, TtlRead

_HIT_FULL = 0.98
"""A cached fraction at or above this renders as a full hit: Claude Code moves the marker."""
_MAX_CELL = 40
"""Longest drift cell rendered before it is elided; the label is planned from the budget."""
_LABEL_FLOOR = 16
"""Narrowest label column: below this a row loses the name it is read by."""
_MIN_DRIFT = 9
"""Narrowest drift column that still shows at least one whole marker."""
_SEPARATOR = " | "
_MAX_TABLE = 120
"""The width both tables are planned to fit: the brief's acceptance line."""
_TTL_LATE_S = 2.0
"""A TTL read later than this past its offset is shown as late, not as exactly on time."""
_PLACEHOLDER = ""
"""Stands in for a label cell while the other columns are measured."""

_RUN_COLUMNS = (
    "spec",
    "hit %",
    "prefix %",
    "eff $/M",
    "in $/M",
    "cold ms",
    "warm ms",
    "TTFT ms",
    "tok/s",
    "errors",
    "drift",
)
_RUNG_COLUMNS = (
    "spec",
    "rung",
    "prompt",
    "cached cold",
    "hits",
    "cold ms",
    "warm ms",
    "ttft ms",
    "tok/s",
    "errors",
)
_TTL_COLUMN = "ttl"


@dataclass(frozen=True, slots=True)
class TableBlock:
    """One table with its columns and its cells already formatted, ready for any renderer."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ProbeBlocks:
    """The caption (when the specs share a prefix) and the two tables, in reading order.

    The cells are the same strings the terminal prints, so a second renderer (the HTML
    report) shows the same numbers without recomputing them; only the layout differs.
    """

    caption: str | None
    spec: TableBlock
    rungs: TableBlock


def render_probe(summaries: Sequence[ProbeSummary]) -> str:
    """The probe's human output: one row per spec, then one row per rung; for a tty."""
    lines = [*_lines(summaries, markdown=False), *probe_note_lines(summaries)]
    return "\n".join(lines) if lines else "\n\n"


def probe_markdown(summaries: Sequence[ProbeSummary]) -> str:
    """The same two tables as markdown, for a report that gets pasted somewhere."""
    lines = [*_lines(summaries, markdown=True), "", *probe_note_lines(summaries)]
    return "\n".join(lines) if lines else "\n\n"


def _lines(summaries: Sequence[ProbeSummary], *, markdown: bool) -> list[str]:
    """Every line of the caption and the two tables, one blank line between blocks."""
    lines: list[str] = []
    for block in _blocks(summaries, markdown=markdown):
        lines.extend(block.splitlines())
        lines.append("")
    return lines[:-1]


def probe_blocks(summaries: Sequence[ProbeSummary]) -> ProbeBlocks:
    """The caption, the spec table and the rung table as data, for any renderer.

    The caption is its own block, which is what `report`'s text extraction splits on to
    recover the two tables whether or not there is one.
    """
    rungs = [rung for summary in summaries for rung in summary.rungs]
    rung_columns = _rung_columns(rungs)
    label_width, drift_width = _planned_widths(summaries, rung_columns)
    caption, labels = _labels_of(summaries, label_width=label_width)
    rows = zip(summaries, labels, strict=True)
    run_rows = [_run_cells(summary, label, drift_width) for summary, label in rows]
    return ProbeBlocks(
        caption=caption,
        spec=TableBlock(columns=_RUN_COLUMNS, rows=tuple(tuple(row) for row in run_rows)),
        rungs=TableBlock(
            columns=rung_columns,
            rows=tuple(tuple(row) for row in _rung_rows(summaries, labels, rung_columns)),
        ),
    )


def _blocks(summaries: Sequence[ProbeSummary], *, markdown: bool) -> list[str]:
    """The optional caption, the spec table and the rung table, rendered as text."""
    blocks = probe_blocks(summaries)
    table = _md_table if markdown else _table
    rendered = [table(block.columns, block.rows) for block in (blocks.spec, blocks.rungs)]
    return [blocks.caption, *rendered] if blocks.caption is not None else rendered


def probe_note_lines(summaries: Sequence[ProbeSummary]) -> list[str]:
    """One `label: note` line per note, in spec order; shared by every renderer."""
    return [f"{summary.label}: {note}" for summary in summaries for note in summary.notes]


def _labels_of(
    summaries: Sequence[ProbeSummary], *, label_width: int
) -> tuple[str | None, list[str]]:
    """The caption and the spec column of every row: the shared decision, on the labels."""
    return column_labels([summary.label for summary in summaries], label_width=label_width)


def _planned_widths(
    summaries: Sequence[ProbeSummary], rung_columns: Sequence[str]
) -> tuple[int, int]:
    """The label and drift widths that hold both tables inside `_MAX_TABLE` columns.

    The label is planned as wide as the budget allows — it is the column a row is looked up
    by, and every column it gets back is one fewer name cut short — and the drift is then
    measured against the label column that actually renders, not against the plan. A shared
    `target:model` in the caption leaves the rows with `@provider` tails, and the columns
    the label column does not use are the drift's rather than nobody's: a drift cell is one
    short marker, `provider,tokens+3%`, and it is worth showing whole.

    A width that would name two rows alike is skipped, and the label is never cut below
    `_LABEL_FLOOR` — past that a readable name beats the column budget. A row wider than
    `_MAX_TABLE` is still possible: a late TTL read writes `(+6)` into the rung table, and
    so does a drift column squeezed under its floor, which is worth the columns too.
    """
    budget = min(
        _MAX_TABLE - _spec_middle(summaries) - _MIN_DRIFT,
        _MAX_TABLE - _rung_span(summaries, rung_columns),
    )
    for width in range(max(budget, _LABEL_FLOOR), _LABEL_FLOOR - 1, -1):
        labels = _labels_of(summaries, label_width=width)[1]
        if named(labels):
            return width, _drift_width(rendered(labels), summaries)
    # nothing fits and names every row; a readable name beats the column budget
    labels = _labels_of(summaries, label_width=_LABEL_FLOOR)[1]
    return _LABEL_FLOOR, _drift_width(rendered(labels), summaries)


def _drift_width(label_width: int, summaries: Sequence[ProbeSummary]) -> int:
    """What is left for drift once the label column and every other column is paid for.

    Never under `_MIN_DRIFT`: a column that cannot show one whole marker is not worth the
    columns it would render in, so a table squeezed past that runs wide instead. This is
    reachable whenever the other columns are wide enough that the label's floor costs more
    than the budget has left for drift.
    """
    leftover = _MAX_TABLE - _spec_middle(summaries) - label_width
    return max(_MIN_DRIFT, min(_MAX_CELL, leftover))


def _spec_middle(summaries: Sequence[ProbeSummary]) -> int:
    """The spec table's width without its label and drift cells, every separator counted."""
    return _columns_span(
        _RUN_COLUMNS,
        [_run_cells(summary, _PLACEHOLDER, _MAX_CELL) for summary in summaries],
        skip=(0, len(_RUN_COLUMNS) - 1),
    )


def _columns_span(
    columns: Sequence[str], rows: Sequence[Sequence[str]], *, skip: Sequence[int]
) -> int:
    """The table's width without the skipped columns: their cells plus every separator."""
    widths = [
        max([len(columns[index]), *(len(row[index]) for row in rows)])
        for index in range(len(columns))
        if index not in skip
    ]
    return sum(widths) + len(_SEPARATOR) * (len(columns) - 1)


def _rung_columns(rungs: Sequence[RungSummary]) -> tuple[str, ...]:
    """The rung columns; `ttl` appears only when some rung carries TTL reads."""
    if any(rung.ttl for rung in rungs):
        return (*_RUNG_COLUMNS[:-1], _TTL_COLUMN, "errors")
    return _RUNG_COLUMNS


def _rung_span(summaries: Sequence[ProbeSummary], rung_columns: Sequence[str]) -> int:
    """The width of the rung table without its label column."""
    rows = _rung_rows(summaries, [""] * len(summaries), rung_columns)
    return _columns_span(rung_columns, rows, skip=(0,))


def _rung_rows(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], columns: Sequence[str]
) -> list[list[str]]:
    """One row per rung, each labelled as its spec row is."""
    column = {summary.label: label for summary, label in zip(summaries, labels, strict=True)}
    with_ttl = _TTL_COLUMN in columns
    return [
        _rung_cells(rung, column.get(rung.spec, rung.spec), with_ttl)
        for summary in summaries
        for rung in summary.rungs
    ]


def _run_cells(summary: ProbeSummary, label: str, drift_width: int) -> list[str]:
    return [
        label,
        _pct(summary.hit_rate),
        _pct(summary.prefix_fraction),
        _money(summary.eff_per_m_prompt, 3),
        _money(summary.input_price, 3),
        _ms(summary.cold_ms),
        _ms(summary.warm_ms),
        _ms(summary.ttft_ms),
        _tok_s(summary.gen_tok_s),
        str(summary.errors),
        elide(summary.drift or "-", drift_width, keep_end=False),
    ]


def _rung_cells(rung: RungSummary, label: str, with_ttl: bool) -> list[str]:
    cells = [
        label,
        str(rung.rung),
        str(rung.prompt_cold),
        str(rung.cached_cold),
        _sequence(rung.hits),
        f"{rung.cold_ms:.0f}",
        _ms(rung.warm_ms),
        _ms(rung.ttft_ms),
        _tok_s(rung.gen_tok_s),
        str(rung.errors),
    ]
    if not with_ttl:
        return cells
    return [*cells[:-1], _ttl_sequence(rung), cells[-1]]


def _sequence(hits: Sequence[float | None]) -> str:
    """`1 1 0 1` per warm attempt: `x` failed, `1` from 0.98 up, else the fraction."""
    return " ".join(_hit_cell(hit) for hit in hits) or "-"


def _ttl_sequence(rung: RungSummary) -> str:
    """`60s:1 300s:0` per `--ttl` read: the offset, then whether the cache was still there.

    A read that went out late says so: `60s:1 (+6)` is a hit measured six seconds past the
    offset it asked for, which is what a slept-through wait or a busy loop looks like.
    """
    if not rung.ttl:
        return "-"
    return " ".join(_ttl_cell(read) for read in rung.ttl)


def _ttl_cell(read: TtlRead) -> str:
    cell = f"{read.offset}s:{int(read.hit)}"
    if read.lateness_s > _TTL_LATE_S:
        return f"{cell} (+{read.lateness_s:.0f})"
    return cell


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


def _tok_s(value: float | None) -> str:
    """Tens of tokens per second is the norm, so one decimal is the useful resolution."""
    return "-" if value is None else f"{value:.1f}"


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """The header, the rule, then one line per row; a table with no rows is only a header."""
    return "\n".join(text_table(columns, rows))


def _md_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
