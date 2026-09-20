"""The report as one self-contained HTML file: `render_html(document)` and nothing else.

The input is the same document `report --json` prints, so the file can be rendered from a
run directory, from a saved document, or in a test with no filesystem at all. The numbers
are not recomputed here: the two probe tables come from `probe_tables.probe_blocks` (reshaped
for the page by `probe_html_tables`) and the full-replay table from `summary.summary_row`,
which is where the terminal gets them from too. This module only decides layout — which is
why a figure in the file and a column in the terminal can never disagree.

The probe page answers two questions a reader who has not opened the README is trying to
answer in a minute: which endpoint to use (`probe_answer.answer_line`, under the title) and
what the difference costs (the `this trace $` column and the closing spend line, both under
"Per provider"). Everything else on the page — the tooltips, the glossary, the caveats
(`probe_caveats`) — exists to make the two tables and the two charts stand on their own for
that reader.

Everything is inlined: the CSS in a `<style>` block, the charts as SVG produced by
`html_svg`, the typeface the system's own stack. There is no script, no image, no font and
no URL, so the file opens offline and renders the same from a USB stick.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html import escape

from provibench.bench.html_style import CSS
from provibench.bench.html_svg import Point, grouped_bars, horizontal_bars, line_chart
from provibench.bench.probe_answer import answer_line
from provibench.bench.probe_caveats import caveats_section, method_line, tok_footnote_cells
from provibench.bench.probe_charts import cost_rows, not_charted, rung_groups, series
from provibench.bench.probe_html_tables import (
    BLANKS_LINE,
    ENDPOINT_HELP,
    GLOSSARY,
    RUNG_HELP,
    dropped_line,
    endpoint_view,
    rung_view,
)
from provibench.bench.probe_summary import ProbeSummary, summary_from_document
from provibench.bench.probe_tables import TableBlock, probe_blocks
from provibench.bench.summary import SUMMARY_COLUMNS, cache_mode_note, summary_row
from provibench.commands.probe_spend import spend_lines
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
    """The title block: what was measured, when, under which cache mode, and (for a probe)
    the one-line method note that used to repeat once per endpoint below."""
    facts = [fact for fact in (_models(document), _mode(document), _created(document)) if fact]
    hex_value = document.get("run_hex")
    if isinstance(hex_value, str) and hex_value:
        facts.append(f"run {hex_value}")
    lines = [f"<h1>{escape(str(document.get('trace')))}</h1>"]
    if facts:
        lines.append(f'<p class="sub">{escape(" · ".join(facts))}</p>')
    lines.append(f'<p class="where">{escape(_relative_run(document.get("run_dir")))}</p>')
    if document.get("protocol") == "probe":
        method = method_line(document)
        if method:
            lines.append(f'<p class="method">{escape(method)}</p>')
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
    """The models the run asked for, deduplicated, in endpoint order.

    The label of an endpoint is `target:model@provider`, which is where both a probe run and a
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
    """Which field a report document carries its per-endpoint summaries under.

    One name for both protocols (a probe run's or a full replay's): `report --json` and
    `probe`/`sweep`/`replay --json` all use `summaries`.
    """
    return "summaries"


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
    """The answer line, the caption, the two probe tables, the two charts and the caveats."""
    summaries = [summary_from_document(entry) for entry in _entries(document, _key(document))]
    blocks = probe_blocks(summaries)
    labels = [row[0] for row in blocks.endpoint.rows]
    sections: list[str] = []

    answer = answer_line(summaries, labels)
    if answer:
        sections.append(f'<p class="answer">{escape(answer)}</p>')
    if blocks.caption:
        sections.append(f'<p class="caption">{escape(blocks.caption)}</p>')

    caveats_html, tok_footnotes = caveats_section(summaries, labels, document)
    endpoint_block, endpoint_dropped = endpoint_view(blocks.endpoint, summaries)
    cell_extra = tok_footnote_cells(endpoint_block, tok_footnotes)
    sections.append(
        _table_section(
            "Per provider",
            endpoint_block,
            caption="Endpoints, cheapest effective prompt price first.",
            help=ENDPOINT_HELP,
            cell_extra=cell_extra,
        )
    )
    # A plain line apiece, not a list: the terminal prints these as closing lines under the
    # table, and one bullet floating under a table reads as an orphaned list item.
    sections.extend(f'<p class="caption">{escape(line)}</p>' for line in spend_lines(document))

    rung_block, rung_dropped, rung_titles = rung_view(blocks.rungs, summaries)
    # `errors` (or any other column) can be constant in both tables at once; name it once.
    dropped = list(dict.fromkeys([*endpoint_dropped, *rung_dropped]))
    if dropped:
        sections.append(f'<p class="caption">{escape(dropped_line(dropped))}</p>')
    sections.append(f'<p class="caption">{GLOSSARY}</p>')
    sections.append(f'<p class="caption">{BLANKS_LINE}</p>')

    sections.append(_table_section("Per rung", rung_block, help=RUNG_HELP, cell_titles=rung_titles))

    charts = _probe_charts(summaries, labels)
    if charts:
        sections.append(f'<section class="charts"><h2>Cache and cost</h2>{charts}</section>')
    if caveats_html:
        sections.append(caveats_html)
    return sections


def _probe_charts(summaries: Sequence[ProbeSummary], labels: Sequence[str]) -> str:
    """The two probe charts, or nothing when no rung has a rate to draw.

    Only the palette's validated slots are charted: past eight series a grouped bar chart
    stops being readable however it is coloured, so the rest stay in the tables above and
    the caption says how many there are. The same eight keep their slot in both charts, so
    an endpoint wears one colour on the page.
    """
    charted = series(summaries, labels)
    groups = rung_groups(charted)
    rows = cost_rows(charted)
    parts: list[str] = []
    unmeasured = not_charted(summaries, labels)
    if groups:
        parts.append(
            "<figure><h3>Share of the prompt served from cache per rung</h3>"
            + _legend([label for _, label in charted])
            + _chart(
                grouped_bars(
                    groups,
                    label="share of the prompt served from cache per rung",
                    y_axis="share %",
                    x_axis="prompt size (rung)",
                ),
                "Share of the prompt served from the cache at each rung's warm reads: `h`, "
                "hit rate times cached fraction, the same quantity `eff $/M` prices from. "
                "The x axis is the probe's rungs, each labelled with the reference "
                "endpoint's cold prompt size in thousands of tokens. Hover a bar for the "
                "endpoint, the rung and the read counts.",
                unmeasured,
            )
            + "</figure>"
        )
    if any(row.value is not None for row in rows):
        parts.append(
            "<figure><h3>Effective prompt price</h3>"
            + _chart(
                horizontal_bars(rows, label="effective prompt price per endpoint"),
                "Effective prompt price per endpoint: USD per 1M prompt tokens at the measured hit "
                "rate, so a cheap listed price that never hits the cache costs more than it "
                "looks. Hover a bar for the listed prices and the hit fraction.",
            )
            + "</figure>"
        )
    return "".join(parts)


def _replay_sections(document: Document) -> list[str]:
    """The per-turn summary table, one cache curve per endpoint, and the notes."""
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


def _curve(entry: Document) -> str:
    """One endpoint's cache curve as a figure, or nothing when the run recorded no turns.

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


def _table_section(
    title: str,
    block: TableBlock,
    *,
    caption: str | None = None,
    help: Mapping[str, str] | None = None,
    cell_titles: Mapping[tuple[int, int], str] | None = None,
    cell_extra: Mapping[tuple[int, int], str] | None = None,
) -> str:
    """One headed table, with an optional caption between the heading and the table itself."""
    cap = f'<p class="caption">{escape(caption)}</p>' if caption else ""
    body = _table(block, help=help, cell_titles=cell_titles, cell_extra=cell_extra)
    return f'<section class="table"><h3>{escape(title)}</h3>{cap}{body}</section>'


def _table(
    block: TableBlock,
    *,
    help: Mapping[str, str] | None = None,
    cell_titles: Mapping[tuple[int, int], str] | None = None,
    cell_extra: Mapping[tuple[int, int], str] | None = None,
) -> str:
    """A scroll container holding one table: the page never scrolls sideways, the table does."""
    help = help or {}
    cell_titles = cell_titles or {}
    cell_extra = cell_extra or {}
    head = "".join(
        f'<th scope="col"{_title_attr(help.get(column))}>{escape(column)}</th>'
        for column in block.columns
    )
    rows = "".join(
        _row(index, row, cell_titles, cell_extra) for index, row in enumerate(block.rows)
    )
    return (
        '<div class="scroll"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _row(
    row_index: int,
    row: Sequence[str],
    cell_titles: Mapping[tuple[int, int], str],
    cell_extra: Mapping[tuple[int, int], str],
) -> str:
    cells: list[str] = []
    for column_index, cell in enumerate(row):
        title = _title_attr(cell_titles.get((row_index, column_index)))
        extra = cell_extra.get((row_index, column_index), "")
        cells.append(f"<td{title}>{escape(cell)}{extra}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def _title_attr(text: str | None) -> str:
    return f' title="{escape(text, quote=True)}"' if text else ""


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
