"""The report as one self-contained HTML file: `render_html(document)` and nothing else.

The input is the same document `report --json` prints, so the file can be rendered from a
run directory, from a saved document, or in a test with no filesystem at all. The numbers
are not recomputed here: the two probe tables come from `probe_tables.probe_blocks` (reshaped
for the page by `probe_html_tables`) and the full-replay table from `summary.summary_row`,
which is where the terminal gets them from too. This module only decides layout — which is
why a figure in the file and a column in the terminal can never disagree.

The probe page answers two questions a reader who has not opened the README is trying to
answer in a minute: which endpoint to use (the endpoint table, sorted cheapest first) and
what the difference costs (the `this trace $` column and the "What a session like this one
bills" chart). Everything else on the page — the tooltips, the method sentence, the caveats
(`probe_caveats`) — exists to make the tables and the charts stand on their own for that
reader.

Everything is inlined: the CSS in a `<style>` block, the charts as SVG produced by
`html_svg`, the typeface the system's own stack. There is no script, no image, no font and
no URL, so the file opens offline and renders the same from a USB stick.

The table and chart markup itself lives in `html_layout`, shared with the full-replay page
below; this module only decides what goes on a page and in what order.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from provibench.bench.html_layout import (
    chart,
    created,
    created_no_seconds,
    footer,
    legend,
    notes,
    relative_run,
    table_section,
    theme_switch,
)
from provibench.bench.html_style import CSS
from provibench.bench.html_svg import Point, grouped_bars, horizontal_bars, line_chart
from provibench.bench.labels import join_and, size_label
from provibench.bench.probe_caveats import caveats_section, run_method_sentence, tok_footnote_cells
from provibench.bench.probe_charts import (
    not_charted,
    price_rows,
    reference_size,
    rung_groups,
    series,
)
from provibench.bench.probe_html_tables import (
    endpoint_caption_html,
    endpoint_help,
    endpoint_view,
    group_header_row,
    rung_help,
    rung_view,
)
from provibench.bench.probe_summary import ProbeSummary, summary_from_document
from provibench.bench.probe_tables import TableBlock, probe_blocks
from provibench.bench.summary import SUMMARY_COLUMNS, cache_mode_note, summary_row
from provibench.core.documents import Document, as_document, as_list

_COUNT_WORDS: dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


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
            theme_switch(),
            _head(document),
            *sections,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _head(document: Document) -> str:
    """The title block: what was measured, when, and (for a probe) at what scale."""
    if document.get("protocol") == "probe":
        return _probe_head(document)
    return _replay_head(document)


def _replay_head(document: Document) -> str:
    facts = [fact for fact in (_models(document), _mode(document), created(document)) if fact]
    lines = [f"<h1>{escape(str(document.get('trace')))}</h1>"]
    if facts:
        lines.append(f'<p class="sub">{escape(" · ".join(facts))}</p>')
    lines.append(f'<p class="where">{escape(relative_run(document.get("run_dir")))}</p>')
    return '<header class="head">' + "".join(lines) + "</header>"


def _probe_head(document: Document) -> str:
    """`Prompt cache check: {model}, {n} endpoints`, then the trace, the date and the run id.

    The cache mode (`cold (nonce)`/`warm`) and the run directory both moved out of the
    header: the mode is now the first caveat, tied to the sentence that explains it, and the
    path is now the footer, since neither helps a reader decide anything up here.
    """
    count = len(_entries(document, _key(document)))
    noun = "endpoint" if count == 1 else "endpoints"
    h1 = f"Prompt cache check: {_probe_model(document)}, {count} {noun}"
    facts = [f"trace {document.get('trace')}"]
    created_at = created_no_seconds(document)
    if created_at:
        facts.append(created_at)
    hex_value = document.get("run_hex")
    if isinstance(hex_value, str) and hex_value:
        facts.append(f"run {hex_value}")
    lines = [f"<h1>{escape(h1)}</h1>", f'<p class="sub">{escape(" · ".join(facts))}</p>']
    return '<header class="head">' + "".join(lines) + "</header>"


def _probe_model(document: Document) -> str:
    """The model the run measured: the sweep's own, or the reference endpoint's."""
    return _sweep_model(document) or _reference_model(_entries(document, _key(document)))


def _sweep_model(document: Document) -> str | None:
    sweep = as_document(document.get("sweep"))
    if sweep is None:
        return None
    model = sweep.get("model")
    return str(model) if model else None


def _reference_model(entries: Sequence[Document]) -> str:
    """The reference endpoint's model, or the first endpoint's when none is marked."""
    reference = next((entry for entry in entries if entry.get("reference") is True), None)
    entry = reference if reference is not None else (entries[0] if entries else None)
    if entry is None:
        return "?"
    label = str(entry.get("label"))
    head = label.split("@", 1)[0]
    return head.partition(":")[2] or head


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


def _probe_sections(document: Document) -> list[str]:
    """The method sentence, the two probe tables, the two charts and the caveats."""
    summaries = [summary_from_document(entry) for entry in _entries(document, _key(document))]
    blocks = probe_blocks(summaries)
    labels = [row[0] for row in blocks.endpoint.rows]
    trace_prompt_tokens = document.get("trace_prompt_tokens")
    sections: list[str] = []

    method = run_method_sentence(summaries, document)
    if method:
        sections.append(f'<p class="method">{escape(method)}</p>')

    caveats_html, tok_footnotes = caveats_section(summaries, labels, document)
    endpoint_block, _endpoint_dropped = endpoint_view(blocks.endpoint, summaries)
    cell_extra = tok_footnote_cells(endpoint_block, tok_footnotes)
    sections.append(
        table_section(
            "Per endpoint",
            endpoint_block,
            caption_html=endpoint_caption_html(
                labels, _probe_model(document), folded=bool(blocks.caption)
            ),
            help=endpoint_help(summaries, trace_prompt_tokens),
            cell_extra=cell_extra,
            group_row=group_header_row(endpoint_block.columns),
            key_column="eff $/M",
        )
    )

    charts = _probe_charts(summaries, labels, document)
    if charts:
        sections.append(charts)

    rung_block, _rung_dropped, rung_titles = rung_view(blocks.rungs, summaries)
    sections.append(
        '<details class="turns"><summary>Per turn: hits and latency</summary>'
        + table_section(None, rung_block, help=rung_help(), cell_titles=rung_titles)
        + "</details>"
    )

    if caveats_html:
        sections.append(caveats_html)
    sections.append(footer(document))
    return sections


def _probe_charts(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], document: Document
) -> str:
    """The price chart, then the cache-share chart, or nothing when neither has data.

    The price chart is the table's `eff $/M` column drawn, one bar per endpoint: the number a
    reader decides by, first and right under the table. The share chart is the evidence for
    it. Nothing else is charted on purpose: the listed prices and the session bill are
    columns of the table, and a second series on the bars was read as a second question.

    Only the palette's validated slots are charted: past eight series a grouped bar chart
    stops being readable however it is coloured, so the rest stay in the tables above and
    the caption says how many there are. The same eight keep their slot in both charts, so
    an endpoint wears one colour on the page.
    """
    charted = series(summaries, labels)
    groups = rung_groups(charted)
    rows = price_rows(charted)
    parts: list[str] = []
    unmeasured = not_charted(summaries, labels)
    if any(row.value is not None for row in rows):
        parts.append(
            "<figure><h3>Price per 1M prompt tokens at the measured hit rate</h3>"
            + chart(
                horizontal_bars(rows, label="eff $/M per endpoint"),
                "The <code>eff $/M</code> column drawn: a cache miss pays the input price, a "
                "hit pays the cache price, weighted by the hit rate this run measured.",
            )
            + "</figure>"
        )
    if groups:
        parts.append(
            "<figure><h3>Share of prompt tokens served from cache, per turn</h3>"
            + legend([label for _, label in charted])
            + chart(
                grouped_bars(
                    groups,
                    label="share of prompt tokens served from cache per turn",
                    y_axis="share %",
                    x_axis="prompt size (turn)",
                ),
                _cache_chart_caption(summaries, document),
                unmeasured,
            )
            + "</figure>"
        )
    return "".join(parts)


def _cache_chart_caption(summaries: Sequence[ProbeSummary], document: Document) -> str:
    caption = (
        "Bar height is hit rate times cached share, the same share <code>eff $/M</code> is "
        "built on. 100 means every repeat hit and the whole prompt was cached."
    )
    tail = _repeat_reads_tail(summaries, document)
    if tail:
        caption += f" {tail}"
    return caption + " Hover a bar for the read counts."


def _repeat_reads_tail(summaries: Sequence[ProbeSummary], document: Document) -> str | None:
    """`At {sizes} each bar is {n} reads, so {100/n} means one miss.`, for the turns that
    were not repeated as many times as the run's most-repeated one."""
    options = as_document(document.get("options")) or {}
    rungs = [value for value in as_list(options.get("rungs")) or [] if isinstance(value, int)]
    repeats = [value for value in as_list(options.get("repeats")) or [] if isinstance(value, int)]
    pairs = list(zip(rungs, repeats, strict=False))
    if not pairs:
        return None
    busiest = max(count for _, count in pairs)
    lighter = [(rung, count) for rung, count in pairs if count < busiest]
    counts = {count for _, count in lighter}
    if len(counts) != 1:
        return None
    [count] = counts
    if count <= 0:
        return None
    sizes = [
        size_label(size) for rung, _ in lighter if (size := reference_size(summaries, rung)) > 0
    ]
    if not sizes:
        return None
    return (
        f"At {join_and(sizes)} each bar is {_count_word(count)} reads, so "
        f"{round(100 / count)} means one of them missed."
    )


def _count_word(count: int) -> str:
    return _COUNT_WORDS.get(count, str(count))


def _replay_sections(document: Document) -> list[str]:
    """The per-turn summary table, one cache curve per endpoint, and the notes."""
    entries = _entries(document, "summaries")
    summary = TableBlock(
        columns=SUMMARY_COLUMNS, rows=tuple(tuple(summary_row(entry)) for entry in entries)
    )
    sections = [table_section("Summary", summary)]
    curves = "".join(_curve(entry) for entry in entries)
    if curves:
        sections.append(f'<section class="charts"><h2>Cache curve</h2>{curves}</section>')
    note_lines = [
        f"{entry.get('label')}: {note}"
        for entry in entries
        for note in map(str, as_list(entry.get("notes")) or [])
    ]
    if note_lines:
        sections.append(notes(note_lines))
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
        f"{chart(line_chart(points, label=label), caption)}</figure>"
    )


def _turn_title(index: int, value: float) -> str:
    """The exact numbers behind one point of a cache curve."""
    return f"turn {index}: {value * 100:.1f}% of the prompt cached"


def _entries(document: Document, key: str = "summaries") -> list[Document]:
    return [d for d in map(as_document, as_list(document.get(key)) or []) if d]
