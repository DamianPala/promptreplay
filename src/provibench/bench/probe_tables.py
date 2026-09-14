"""The probe's two tables: one row per spec, one row per rung, as text or markdown.

Kept apart from `probe_summary` (which folds the records into the numbers) because the
rendering has its own problems: the spec column has to stay readable when every spec shares
the `target:model` prefix without ever clipping two specs into the same string, a column
that only exists when `--ttl` ran must not leave a hole when it did not, and the whole
thing has to fit 120 columns.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.probe_summary import ProbeSummary, RungSummary, TtlRead

_HIT_FULL = 0.98
"""A cached fraction at or above this renders as a full hit: Claude Code moves the marker."""
_MAX_CELL = 40
"""Longest label or drift cell rendered before it is elided."""
_LABEL_FLOOR = 16
"""Narrowest label column: below this a row loses the name it is read by."""
_MIN_DRIFT = 9
"""Narrowest drift column that still shows at least one whole marker."""
_ELLIPSIS = "…"
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


def render_probe(summaries: Sequence[ProbeSummary]) -> str:
    """The probe's human output: one row per spec, then one row per rung; for a tty."""
    lines = [*_lines(summaries, markdown=False), *_note_lines(summaries)]
    return "\n".join(lines) if lines else "\n\n"


def probe_markdown(summaries: Sequence[ProbeSummary]) -> str:
    """The same two tables as markdown, for a report that gets pasted somewhere."""
    lines = [*_lines(summaries, markdown=True), "", *_note_lines(summaries)]
    return "\n".join(lines) if lines else "\n\n"


def _lines(summaries: Sequence[ProbeSummary], *, markdown: bool) -> list[str]:
    """Every line of the caption and the two tables, one blank line between blocks."""
    lines: list[str] = []
    for block in _blocks(summaries, markdown=markdown):
        lines.extend(block.splitlines())
        lines.append("")
    return lines[:-1]


def _blocks(summaries: Sequence[ProbeSummary], *, markdown: bool) -> list[str]:
    """The optional caption, the spec table and the rung table, in that order.

    The caption is its own block, which is what `report`'s text extraction splits on to
    recover the two tables whether or not there is one.
    """
    rungs = [rung for summary in summaries for rung in summary.rungs]
    rung_columns = _rung_columns(rungs)
    label_width, drift_width = _planned_widths(summaries, rung_columns)
    caption, labels = column_labels(summaries, label_width=label_width)
    rows = zip(summaries, labels, strict=True)
    run_rows = [_run_cells(summary, label, drift_width) for summary, label in rows]
    rung_rows = _rung_rows(summaries, labels, rung_columns)
    table = _md_table if markdown else _table
    blocks = [table(_RUN_COLUMNS, run_rows), table(rung_columns, rung_rows)]
    return [caption, *blocks] if caption is not None else blocks


def column_labels(
    summaries: Sequence[ProbeSummary], *, label_width: int = _MAX_CELL
) -> tuple[str | None, list[str]]:
    """The caption and the spec column of every row, as one shared decision.

    A `target:model` that at least two specs share moves into the caption, so those rows
    show their provider tails (`@novita`, `@gmicloud`) while a spec of another model keeps
    its own name. Both tables take the result, so a rung row is readable next to its spec
    row. The caption may name several shared prefixes; when nothing is shared there is no
    caption and every row keeps its whole label. A shortened set that repeats is thrown
    away: two rows reading the same string name neither of them.
    """
    labels = [summary.label for summary in summaries]
    heads = [_head(label) for label in labels]
    shared = sorted({head for head in heads if head and heads.count(head) > 1})
    column = [
        _short_label(label, head, shared, label_width)
        for label, head in zip(labels, heads, strict=True)
    ]
    if len(set(column)) != len(column):
        shared = []
        column = [_elide(label, label_width) for label in labels]
    caption = "specs: " + ", ".join(f"{head}@<provider>" for head in shared) if shared else None
    return caption, column


def _head(label: str) -> str:
    """A label's `target:model` part, without its `@provider` suffix."""
    head, at, _ = label.rpartition("@")
    return head if at else label


def _short_label(label: str, head: str, shared: Sequence[str], width: int) -> str:
    """`label` without the shared head, or whole when this spec's model is its own."""
    tail = label[len(head) :] if head in shared else ""
    return _elide(tail or label, width)


def _elide(cell: str, width: int, *, keep_end: bool = True) -> str:
    """`cell` cut to `width` with the middle dropped: `openrouter:deepseek/…@novita`.

    A label's provider suffix is its shortest distinguishing part, so it survives whole and
    the head keeps what is left. A label without one keeps both of its ends instead: its
    model sits at the end and its target at the start, and cutting either loses the pair of
    names that tell one row from another. `keep_end=False` is for a cell whose start is what
    matters — a comma-separated list of drift markers — and cuts its tail.
    """
    if len(cell) <= width:
        return cell
    if not keep_end:
        return f"{cell[: width - len(_ELLIPSIS)]}{_ELLIPSIS}"
    at = cell.rfind("@")
    if at > 0:
        tail = cell[at:]
        keep = width - len(tail) - len(_ELLIPSIS)
        if keep >= 1:
            return f"{cell[:keep]}{_ELLIPSIS}{tail}"
    head = max(1, (width - len(_ELLIPSIS)) // 3)
    return f"{cell[:head]}{_ELLIPSIS}{cell[-(width - len(_ELLIPSIS) - head) :]}"


def _planned_widths(
    summaries: Sequence[ProbeSummary], rung_columns: Sequence[str]
) -> tuple[int, int]:
    """The label and drift widths that hold both tables inside `_MAX_TABLE` columns.

    The spec table's own columns need most of the width, so the label and drift split what
    is left. The label is as wide as that budget allows, because the wider it is the more of
    each row's name survives; the drift takes the rest, and the label narrows — but no
    further than `_LABEL_FLOOR` — until the drift can show a whole marker and both tables
    fit. A width that would name two rows alike is skipped.

    A row wider than `_MAX_TABLE` is still possible: a late TTL read writes `(+6)` into the
    rung table, and that is worth the extra columns. The label is never cut to pay for it.
    """
    room = min(
        _MAX_TABLE - _drift_width(0, summaries),
        _MAX_TABLE - _rung_span(summaries, rung_columns),
    )
    for width in range(min(_MAX_CELL, max(room, _LABEL_FLOOR)), _LABEL_FLOOR - 1, -1):
        drift = _drift_width(width, summaries)
        if drift < _MIN_DRIFT:
            continue
        if _named(column_labels(summaries, label_width=width)[1]):
            return width, drift
    # nothing fits and names every row; a readable name beats the column budget
    return _LABEL_FLOOR, max(_MIN_DRIFT, _drift_width(_LABEL_FLOOR, summaries))


def _named(labels: Sequence[str]) -> bool:
    """Whether every row still has a name of its own after the cut."""
    return len(set(labels)) == len(labels)


def _drift_width(label_width: int, summaries: Sequence[ProbeSummary]) -> int:
    """What is left for drift once the label column and every other column is paid for."""
    fixed = _columns_span(
        _RUN_COLUMNS,
        [_run_cells(summary, _PLACEHOLDER, _MAX_CELL) for summary in summaries],
        skip=(0, len(_RUN_COLUMNS) - 1),
    )
    return min(_MAX_CELL, _MAX_TABLE - fixed - label_width)


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


def _note_lines(summaries: Sequence[ProbeSummary]) -> list[str]:
    return [f"{summary.label}: {note}" for summary in summaries for note in summary.notes]


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
        _elide(summary.drift or "-", drift_width, keep_end=False),
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
