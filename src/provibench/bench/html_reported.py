"""The HTML page's section comparing a run against OpenRouter's reported average.

Drawn by hand rather than through `html_layout.table_section`/`_table`: that renderer keys a
column's tooltip by its header text, and this table's three `OR avg` columns would collide on
one tooltip if it did. The alternative -- teaching `_table` to key tooltips by column index,
or giving the three columns distinct internal names it never shows -- touches code the other
probe-page tables already depend on for a table this section alone renders; drawing the few
rows here, in the same `section`/`scroll`/`table`/`key` classes the rest of the page uses, is
the smaller change.

The cell values themselves are not recomputed: `reported_average.reported_average_row` is the
same function the text and Markdown tables call, so a number here and in `report --format
text` can never disagree.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from html import escape

from provibench.bench.probe_summary import ProbeSummary
from provibench.bench.reported_average import (
    pooled_note,
    reported_average_caption,
    reported_average_row,
    reported_average_sentence,
)
from provibench.core.documents import Document, as_document

__all__ = ["reported_average_section"]

_GROUPS = ("prompt tokens from cache %", "input $/M", "this trace, input tokens")

_SHARE_TOOLTIPS = (
    "Share of the prompt tokens this run's repeat requests had served from cache: hit % "
    "times cached %, the h behind eff $/M.",
    "Share of prompt tokens served from cache across every OpenRouter customer of this "
    "endpoint on {day}, as OpenRouter's model page reports it.",
)
_PRICE_TOOLTIPS = (
    "Input price per 1M prompt tokens at this run's cache share: the table's eff $/M.",
    "The same arithmetic with OpenRouter's reported share in place of this run's, at this "
    "row's listed prices. Not OpenRouter's own observed price, which folds in discounts.",
)
_BILL_TOOLTIPS = (
    "The table's this trace $: the recorded session's input tokens at eff $/M.",
    "The recorded session's input tokens at the OR avg price.",
)
_VS_TOOLTIP = (
    "This run's bill against the OR avg bill on the same endpoint. Negative: this session's "
    "prompts cache better here than OpenRouter's average traffic does."
)


_CODE_SPAN = re.compile(r"`([^`]+)`")


def _code_spans(escaped: str) -> str:
    """Backtick spans in the shared caption become `<code>` on the page.

    The caption is one string for the text, Markdown and HTML renderers, and names the
    `vs OR avg` column the way the text report does, in backticks; the page shows column
    names in `<code>` everywhere else, so raw backticks here would be the one exception.
    """
    return _CODE_SPAN.sub(r"<code>\1</code>", escaped)


def reported_average_section(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], document: Document
) -> str | None:
    """`None` when nothing has a figure, or the run's day is unknown."""
    day = _day(document)
    pairs = [
        (summary, label)
        for summary, label in zip(summaries, labels, strict=True)
        if summary.or_avg_share_pct is not None
    ]
    if day is None or not pairs:
        return None
    trace_tokens = document.get("trace_prompt_tokens")
    caption = reported_average_caption(day, trace_tokens if isinstance(trace_tokens, int) else None)
    caption_html = _code_spans(escape(caption))
    if any(summary.or_avg_pooled for summary, _ in pairs):
        caption_html += f' <span class="hint">{escape(pooled_note())}</span>'
    sentence = reported_average_sentence(summaries, labels)
    tail = f'<p class="caption">{escape(sentence)}</p>' if sentence else ""
    return (
        '<section class="table"><h3>Against OpenRouter\'s reported average</h3>'
        f'<p class="caption">{caption_html}</p>'
        f'<div class="scroll">{_table(pairs, day)}</div>{tail}</section>'
    )


def _day(document: Document) -> str | None:
    reported = as_document(document.get("reported_average"))
    day = reported.get("day") if reported is not None else None
    return day if isinstance(day, str) else None


def _table(pairs: Sequence[tuple[ProbeSummary, str]], day: str) -> str:
    groups = "".join(
        f'<th scope="colgroup" colspan="2" class="group-start">{escape(name)}</th>'
        for name in _GROUPS
    )
    head = f'<tr class="groups"><th></th>{groups}<th></th></tr><tr>{_header_cells(day)}</tr>'
    rows = "".join(_row(summary, label) for summary, label in pairs)
    return f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>"


def _header_cells(day: str) -> str:
    cells = ['<th scope="col">endpoint</th>']
    for pair_index, tooltips in enumerate((_SHARE_TOOLTIPS, _PRICE_TOOLTIPS, _BILL_TOOLTIPS)):
        for column_index, tooltip in enumerate(tooltips):
            text = tooltip.format(day=day)
            key = ' class="key"' if pair_index == 1 and column_index == 0 else ""
            word = "this run" if column_index == 0 else "OR avg"
            cells.append(f'<th scope="col"{key} title="{escape(text, quote=True)}">{word}</th>')
    cells.append(f'<th scope="col" title="{escape(_VS_TOOLTIP, quote=True)}">vs OR avg</th>')
    return "".join(cells)


_KEY_COLUMN = 3
"""`reported_average_row`'s `eff $/M` column for this run: the number the page highlights."""


def _row(summary: ProbeSummary, label: str) -> str:
    cells = reported_average_row(summary, label)
    tds = [
        f'<td class="key">{escape(cell)}</td>'
        if index == _KEY_COLUMN
        else f"<td>{escape(cell)}</td>"
        for index, cell in enumerate(cells)
    ]
    return "<tr>" + "".join(tds) + "</tr>"
