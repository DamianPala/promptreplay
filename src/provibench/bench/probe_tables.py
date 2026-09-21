"""The probe's two tables: one row per endpoint, one row per rung, as text or markdown.

Kept apart from `probe_summary` (which folds the records into the numbers) because the
rendering has its own problems: the endpoint column has to stay readable when every
endpoint shares the `target:model` prefix without ever clipping two endpoints into the
same string, a column that only exists when `--ttl` ran must not leave a hole when it did
not, and the whole thing has to fit 120 columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from provibench.bench.labels import column_labels, named, text_table
from provibench.bench.probe_summary import ProbeSummary, RungSummary, TtlRead
from provibench.bench.summary import cache_mode_note, cache_mode_sentence

_HIT_FULL = 0.98
"""A cached fraction at or above this renders as a full hit: Claude Code moves the marker."""
_LABEL_FLOOR = 32
"""Narrowest label column: below this a row loses the name it is read by. Wide enough that a
provider tag like `@sail-research/fp8` or `@sail-research/us` survives whole; there is
plenty of room in a table capped at `_MAX_TABLE`."""
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
    "endpoint",
    "hit %",
    "1st hit %",
    "cached %",
    "in $/M",
    "cache $/M",
    "eff $/M",
    "cold ms",
    "warm ms",
    "TTFT ms",
    "tok/s",
    "errors",
    "drift",
)
_RUNG_COLUMNS = (
    "endpoint",
    "rung",
    "prompt",
    "cached cold",
    "hits",
    "cold ms",
    "warm ms",
    # `TTFT` is an acronym and the endpoint table, `sweep` and `compare` all spell it in
    # caps; the two tables sat side by side spelling the same measurement two ways.
    "TTFT ms",
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
    """The caption (when the endpoints share a prefix) and the two tables, in reading order.

    The cells are the same strings the terminal prints, so a second renderer (the HTML
    report) shows the same numbers without recomputing them; only the layout differs.
    """

    caption: str | None
    endpoint: TableBlock
    rungs: TableBlock
    reported: TableBlock | None  # vs OpenRouter's reported average, when a summary carries one
    reported_caption: str | None  # caption above `reported`; `None` without `reported` or `day`


def render_probe(summaries: Sequence[ProbeSummary], *, day: str | None = None) -> str:
    """The probe's human output: one row per endpoint, then one row per rung; for a tty."""
    return _render(summaries, markdown=False, day=day)


def probe_markdown(summaries: Sequence[ProbeSummary], *, day: str | None = None) -> str:
    """The same two tables as markdown, for a report that gets pasted somewhere."""
    return _render(summaries, markdown=True, day=day)


def probe_labels(summaries: Sequence[ProbeSummary]) -> list[str]:
    """The short label each endpoint gets in the tables above, for a caller that renders its
    own lines (the closing `prompt bill`) and wants to name the same endpoint the same way."""
    return _labels_from(probe_blocks(summaries))


def _render(summaries: Sequence[ProbeSummary], *, markdown: bool, day: str | None = None) -> str:
    blocks = probe_blocks(summaries, day=day)
    lines = _lines(blocks, markdown=markdown)
    notes = probe_note_lines(summaries, _labels_from(blocks), day=day)
    if notes:
        lines = [*lines, "", *notes]
    return "\n".join(lines) if lines else "\n\n"


def _labels_from(blocks: ProbeBlocks) -> list[str]:
    """The label column of the endpoint table, in row order: the short form it already shows."""
    return [row[0] for row in blocks.endpoint.rows]


def _lines(blocks: ProbeBlocks, *, markdown: bool) -> list[str]:
    """Every line of the caption and the two tables, one blank line between blocks."""
    lines: list[str] = []
    for block in _blocks(blocks, markdown=markdown):
        lines.extend(block.splitlines())
        lines.append("")
    return lines[:-1]


def probe_blocks(summaries: Sequence[ProbeSummary], *, day: str | None = None) -> ProbeBlocks:
    """The caption, the endpoint table and the rung table as data, for any renderer.

    The caption is its own block, which is what `report`'s text extraction splits on to
    recover the two tables whether or not there is one. `day` -- the run's own day, known
    only to a caller that has the full report document -- is what lets the OR-avg caption
    name the date; without it the comparison table still renders, captionless.
    """
    from provibench.bench.reported_average import (
        reported_average_block,
        reported_average_table_caption,
    )

    rungs = [rung for summary in summaries for rung in summary.rungs]
    rung_columns = _rung_columns(rungs)
    label_width = _planned_widths(summaries, rung_columns)
    caption, labels = _labels_of(summaries, label_width=label_width)
    rows = zip(summaries, labels, strict=True)
    run_rows = [_run_cells(summary, label) for summary, label in rows]
    reported = reported_average_block(summaries, labels)
    return ProbeBlocks(
        caption=caption,
        endpoint=TableBlock(columns=_RUN_COLUMNS, rows=tuple(tuple(row) for row in run_rows)),
        rungs=TableBlock(
            columns=rung_columns,
            rows=tuple(tuple(row) for row in _rung_rows(summaries, labels, rung_columns)),
        ),
        reported=reported,
        reported_caption=reported_average_table_caption(day)
        if reported is not None and day
        else None,
    )


def _blocks(blocks: ProbeBlocks, *, markdown: bool) -> list[str]:
    """The optional caption, the two tables, then the OR-avg caption and table when present."""
    table = _md_table if markdown else _table
    rendered = [table(block.columns, block.rows) for block in (blocks.endpoint, blocks.rungs)]
    parts = [blocks.caption, *rendered] if blocks.caption is not None else rendered
    if blocks.reported is not None:
        if blocks.reported_caption is not None:
            parts = [*parts, blocks.reported_caption]
        parts = [*parts, table(blocks.reported.columns, blocks.reported.rows)]
    return parts


_CACHE_MODE_NOTES = frozenset({cache_mode_note(True), cache_mode_note(False)})


def probe_note_lines(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], *, day: str | None = None
) -> list[str]:
    """The run-wide cache-mode sentence once, unprefixed, one `{label}: {note}` line per
    remaining note, then the OR-avg pooled note and source sentence when they apply -- shared
    by every renderer. `labels` is the endpoint table's own short column
    (`probe_blocks(...).endpoint.rows`, or `probe_labels`): a note names the same endpoint the
    table above it does, `@relace/fp4` rather than the full label the caption already expanded.
    """
    shared = _shared_cache_mode(summaries)
    lines = [cache_mode_sentence(shared == cache_mode_note(True))] if shared is not None else []
    lines.extend(
        f"{label}: {note}"
        for summary, label in zip(summaries, labels, strict=True)
        for note in summary.notes
        if note != shared
    )
    lines.extend(_reported_average_notes(summaries, day))
    return lines


def _reported_average_notes(summaries: Sequence[ProbeSummary], day: str | None) -> list[str]:
    from provibench.bench.reported_average import pooled_note, reported_average_source

    notes: list[str] = []
    if any(summary.or_avg_pooled for summary in summaries):
        notes.append(pooled_note())
    if day is not None and any(summary.or_avg_share_pct is not None for summary in summaries):
        notes.append(reported_average_source(day))
    return notes


def _shared_cache_mode(summaries: Sequence[ProbeSummary]) -> str | None:
    """The cache-mode note every summary carries alike, or `None` when there is no such note.

    `cache_mode_note` is a fact about the run, not about one endpoint, so `probe`/`report`
    write it into every summary's notes the same way; a run of one endpoint still shares it
    with itself, and a summary with no notes at all (a dry construction in a test) has none.
    """
    if not summaries:
        return None
    candidates = {note for note in summaries[0].notes if note in _CACHE_MODE_NOTES}
    if len(candidates) != 1:
        return None
    (note,) = candidates
    return note if all(note in summary.notes for summary in summaries[1:]) else None


def _labels_of(
    summaries: Sequence[ProbeSummary], *, label_width: int
) -> tuple[str | None, list[str]]:
    """The caption and the endpoint column of every row: the shared decision, on the labels."""
    return column_labels([summary.label for summary in summaries], label_width=label_width)


def _planned_widths(summaries: Sequence[ProbeSummary], rung_columns: Sequence[str]) -> int:
    """The label width that holds both tables inside `_MAX_TABLE` columns, drift aside.

    The label is planned as wide as the budget allows — it is the column a row is looked up
    by, and every column it gets back is one fewer name cut short. The drift column is never
    truncated (see `_run_cells`): a marker like `provider,tokens+3%` is short and worth
    showing whole even when that runs the table past `_MAX_TABLE`, so only `_MIN_DRIFT`
    columns are reserved for it here, as a floor under the label's own budget.

    A width that would name two rows alike is skipped, and the label is never cut below
    `_LABEL_FLOOR` — past that a readable name beats the column budget. A row wider than
    `_MAX_TABLE` is still possible: a late TTL read writes `(+6)` into the rung table, a
    drift marker longer than the floor, or a dense table whose spare width falls under the
    floor, are all worth the columns they cost.
    """
    spare = _MAX_TABLE - _endpoint_middle(summaries) - _MIN_DRIFT
    budget = min(spare, _MAX_TABLE - _rung_span(summaries, rung_columns))
    for width in range(max(budget, _LABEL_FLOOR), _LABEL_FLOOR - 1, -1):
        labels = _labels_of(summaries, label_width=width)[1]
        if named(labels):
            return width
    # nothing fits and names every row; a readable name beats the column budget
    return _LABEL_FLOOR


def _endpoint_middle(summaries: Sequence[ProbeSummary]) -> int:
    """The endpoint table's width without its label and drift cells, every separator counted."""
    return _columns_span(
        _RUN_COLUMNS,
        [_run_cells(summary, _PLACEHOLDER) for summary in summaries],
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
    """One row per rung, each labelled as its endpoint row is."""
    column = {summary.label: label for summary, label in zip(summaries, labels, strict=True)}
    with_ttl = _TTL_COLUMN in columns
    return [
        _rung_cells(rung, column.get(rung.endpoint, rung.endpoint), with_ttl)
        for summary in summaries
        for rung in summary.rungs
    ]


def _run_cells(summary: ProbeSummary, label: str) -> list[str]:
    return [
        label,
        _pct(summary.hit_rate),
        _pct(summary.first_hit_rate),
        _pct(summary.cached_fraction),
        _money(summary.input_price, 3),
        _money(summary.cache_read_price, 3),
        _money(summary.eff_per_m_prompt, 3),
        _ms(summary.cold_ms),
        _ms(summary.warm_ms),
        _ms(summary.ttft_ms),
        _tok_s(summary.gen_tok_s),
        str(summary.errors),
        # Never elided: a drift marker (`provider,tokens+3%`) is short and worth showing
        # whole, even at the cost of a table wider than `_MAX_TABLE`.
        summary.drift or "-",
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
    cell = f"{read.offset_s}s:{int(read.hit)}"
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
