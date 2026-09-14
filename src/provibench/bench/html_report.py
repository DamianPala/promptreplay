"""The report as one self-contained HTML file: `render_html(document)` and nothing else.

The input is the same document `report --json` prints, so the file can be rendered from a
run directory, from a saved document, or in a test with no filesystem at all. The numbers
are not recomputed here: the two probe tables come from `probe_tables.probe_blocks` and the
full-replay table from `summary.summary_row`, which is where the terminal gets them from
too. This module only decides layout — which is why a figure in the file and a column in
the terminal can never disagree.

Everything is inlined: the CSS in a `<style>` block, the charts as SVG produced by
`html_svg`, the typeface the system's own stack. There is no script, no image, no font and
no URL, so the file opens offline and renders the same from a USB stick.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from html import escape

from provibench.bench.html_style import CSS
from provibench.bench.html_svg import Point, grouped_bars, horizontal_bars, line_chart
from provibench.bench.probe_charts import cost_rows, not_charted, rung_groups, series
from provibench.bench.probe_summary import ProbeSummary, summary_from_document
from provibench.bench.probe_tables import TableBlock, probe_blocks, probe_note_lines
from provibench.bench.selection import SweepInfo, not_probed_lines, selection_line
from provibench.bench.summary import SUMMARY_COLUMNS, cache_mode_note, summary_row
from provibench.core.documents import Document, as_document, as_list

_STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$")


def render_html(document: Document) -> str:
    """One HTML document showing `document`, exactly as `report --json` carries it."""
    sections = (
        _probe_sections(document)
        if document.get("protocol") == "probe"
        else _replay_sections(document)
    )
    trace = str(document.get("trace"))
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>provibench report: {escape(trace)}</title>",
            f"<style>\n{CSS}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            _head(document),
            *sections,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _head(document: Document) -> str:
    """The title block: what was measured, when, and under which cache mode."""
    facts = [fact for fact in (_models(document), _mode(document), _created(document)) if fact]
    hex_value = document.get("run_hex")
    if isinstance(hex_value, str) and hex_value:
        facts.append(f"run {hex_value}")
    lines = [f"<h1>{escape(str(document.get('trace')))}</h1>"]
    if facts:
        lines.append(f'<p class="sub">{escape(" · ".join(facts))}</p>')
    lines.append(f'<p class="where">{escape(_relative_run(document.get("run_dir")))}</p>')
    return '<header class="head">' + "".join(lines) + "</header>"


def _relative_run(run_dir: object) -> str:
    """The run as `runs/<trace>/<timestamp>`: where it is, without the operator's paths.

    A report is meant to be shared, and an absolute path names the machine it was made on.
    The last two segments are the run's identity — the trace and its timestamp — and the
    `runs/` prefix says what they are relative to.
    """
    segments = [part for part in str(run_dir).replace("\\", "/").split("/") if part]
    return "/".join(["runs", *segments[-2:]])


def _models(document: Document) -> str:
    """The models the run asked for, deduplicated, in spec order.

    The label of a spec is `target:model@provider`, which is where both a probe run and a
    full replay keep the model they measured; the model a response named is a different
    claim, and it is the drift column's job, not the title's.
    """
    models: list[str] = []
    for entry in _entries(document, _key(document)):
        label = str(entry.get("label"))
        head = label.split("@", 1)[0]
        model = head.partition(":")[2] or head
        if model and model not in models:
            models.append(model)
    return ", ".join(models)


def _key(document: Document) -> str:
    """Which summary list a document carries: a probe run's, or a full replay's."""
    return "probe_summaries" if document.get("protocol") == "probe" else "summaries"


def _mode(document: Document) -> str:
    """`cold (nonce)` or `warm`, as the terminal's notes name them."""
    options = as_document(document.get("options")) or {}
    warm = options.get("warm")
    return cache_mode_note(warm) if isinstance(warm, bool) else ""


def _created(document: Document) -> str:
    """The run's timestamp, spaced out when it is the `run.json` stamp."""
    created = str(document.get("created"))
    stamp = _STAMP.match(created)
    if stamp is None:
        return created
    date, time = stamp.group(1, 2, 3), stamp.group(4, 5, 6)
    return f"{date[0]}-{date[1]}-{date[2]} {time[0]}:{time[1]}:{time[2]}"


def _probe_sections(document: Document) -> list[str]:
    """The caption, the two probe tables, the two charts and the notes."""
    summaries = [summary_from_document(entry) for entry in _entries(document, "probe_summaries")]
    blocks = probe_blocks(summaries)
    labels = [row[0] for row in blocks.spec.rows]
    sections: list[str] = []
    if blocks.caption:
        sections.append(f'<p class="caption">{escape(blocks.caption)}</p>')
    sections.append(_table_section("Per provider", blocks.spec))
    sections.append(_table_section("Per rung", blocks.rungs))
    charts = _probe_charts(summaries, labels)
    if charts:
        sections.append(f'<section class="charts"><h2>Cache and cost</h2>{charts}</section>')
    notes = [*probe_note_lines(summaries), *_selection_lines(document)]
    if notes:
        sections.append(_notes(notes))
    return sections


def _selection_lines(document: Document) -> list[str]:
    """A sweep's selection as note lines, or nothing for a probe whose specs were named.

    The report is the artifact that gets shared, so the criteria the run chose by and the
    endpoints they dropped belong in the file next to the numbers they explain.
    """
    block = as_document(document.get("sweep"))
    if block is None:
        return []
    sweep = SweepInfo.model_validate(block)
    return [selection_line(sweep), *not_probed_lines(sweep)]


def _replay_sections(document: Document) -> list[str]:
    """The per-turn summary table, one cache curve per spec, and the notes."""
    entries = _entries(document, "summaries")
    summary = TableBlock(
        columns=SUMMARY_COLUMNS, rows=tuple(tuple(summary_row(entry)) for entry in entries)
    )
    sections = [_table(summary)]
    curves = "".join(_curve(entry) for entry in entries)
    if curves:
        sections.append(f'<section class="charts"><h2>Cache curve</h2>{curves}</section>')
    notes = [
        f"{entry.get('label')}: {note}"
        for entry in entries
        for note in map(str, as_list(entry.get("notes")) or [])
    ]
    if notes:
        sections.append(_notes(notes))
    return sections


def _probe_charts(summaries: Sequence[ProbeSummary], labels: Sequence[str]) -> str:
    """The two probe charts, or nothing when no rung has a rate to draw.

    Only the palette's validated slots are charted: past eight series a grouped bar chart
    stops being readable however it is coloured, so the rest stay in the tables above and
    the caption says how many there are. The same eight keep their slot in both charts, so
    a spec wears one colour on the page.
    """
    charted = series(summaries, labels)
    groups = rung_groups(charted)
    rows = cost_rows(charted)
    parts: list[str] = []
    unmeasured = not_charted(summaries, labels)
    if groups:
        parts.append(
            f"<figure>{_legend([label for _, label in charted])}"
            + _chart(
                grouped_bars(groups, label="hit rate per rung", y_axis="hit %", x_axis="rung"),
                "Hit rate of the warm reads at each rung. The x axis is the probe's rungs, "
                "each labelled with the reference spec's cold prompt size in thousands of "
                "tokens; the height is the share of the served warm reads whose prompt was "
                "already cached. Hover a bar for the spec, the rung and the read counts.",
                unmeasured,
            )
            + "</figure>"
        )
    if any(row.value is not None for row in rows):
        parts.append(
            "<figure><h3>Effective prompt price</h3>"
            + _chart(
                horizontal_bars(rows, label="effective prompt price per spec"),
                "Effective prompt price per spec: USD per 1M prompt tokens at the measured hit "
                "rate, so a cheap listed price that never hits the cache costs more than it "
                "looks. Hover a bar for the listed prices and the hit fraction.",
            )
            + "</figure>"
        )
    return "".join(parts)


def _curve(entry: Document) -> str:
    """One spec's cache curve as a figure, or nothing when the run recorded no turns.

    A fraction outside 0..1 is clamped, as the terminal's sparkline clamps it: the curve is
    read as "how much of the prompt was cached", and a value the axis cannot hold would put
    the point and the number hovering it in different places.
    """
    values = [
        min(1.0, max(0.0, float(v)))
        for v in as_list(entry.get("curve")) or []
        if isinstance(v, int | float)
    ]
    if not values:
        return ""
    label = str(entry.get("label"))
    points = [
        Point(index, value, _turn_title(index, value))
        for index, value in enumerate(values, start=1)
    ]
    caption = f"Cached fraction of each turn's prompt across the replay's {len(values)} turns."
    return (
        f"<figure><h3>{escape(label)}</h3>"
        f"{_chart(line_chart(points, label=label), caption)}</figure>"
    )


def _turn_title(index: int, value: float) -> str:
    """The exact numbers behind one point of a cache curve."""
    return f"turn {index}: {value * 100:.1f}% of the prompt cached"


def _table_section(title: str, block: TableBlock) -> str:
    """One headed table: the probe has two, and without a heading they read as one."""
    return f'<section class="table"><h3>{escape(title)}</h3>{_table(block)}</section>'


def _table(block: TableBlock) -> str:
    """A scroll container holding one table: the page never scrolls sideways, the table does."""
    head = "".join(f'<th scope="col">{escape(column)}</th>' for column in block.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in block.rows
    )
    return (
        '<div class="scroll"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _legend(labels: Sequence[str]) -> str:
    items = "".join(
        f'<li><span class="swatch s{index + 1}"></span>{escape(label)}</li>'
        for index, label in enumerate(labels)
    )
    return f'<ul class="legend">{items}</ul>'


def _chart(svg: str, caption: str, unmeasured: Sequence[str] = ()) -> str:
    """The scroll container, why some rungs have no bar, then the caption."""
    aside = f'<p class="caption">Not charted</p>{_notes(unmeasured)}' if unmeasured else ""
    return f'<div class="chart-scroll">{svg}</div>{aside}<figcaption>{escape(caption)}</figcaption>'


def _notes(lines: Sequence[str]) -> str:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    return f'<ul class="notes">{items}</ul>'


def _entries(document: Document, key: str = "summaries") -> list[Document]:
    return [d for d in map(as_document, as_list(document.get(key)) or []) if d]
