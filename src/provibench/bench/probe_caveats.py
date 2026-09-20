"""The probe page's Caveats section, and the run-wide method note shown once above it.

`ProbeSummary.notes` carries the run-wide cache-mode note on every endpoint alike (`probe`
puts it there so the text report shows it per row); the HTML page instead prints it once,
under the header, and turns every other note into a `<li>` under its own short, bold
endpoint label -- never the full spec string a reader has not seen anywhere else on the
page. A burst note also earns a footnote link from the blank `tok/s` cell it explains.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from provibench.bench.probe_summary import ProbeSummary
from provibench.bench.probe_tables import TableBlock
from provibench.bench.selection import SweepInfo, not_probed_lines, selection_line
from provibench.bench.summary import cache_mode_note
from provibench.core.documents import Document, as_document

__all__ = ["caveats_section", "method_line", "tok_footnote_cells"]

_METHOD_LINES: dict[bool, str] = {
    False: (
        "Every cache hit below was written by this run: each request carried a unique "
        "marker, so nothing was inherited from earlier traffic."
    ),
}
"""The run-wide cache-mode note, written once under the header instead of once per endpoint.
Keyed by `options.warm`; a warm run has no rewritten wording to show here (the report's own
review only rewrote the cold case), so it keeps its short form in the header subtitle alone."""

_METHOD_NOTES = (cache_mode_note(True), cache_mode_note(False))
"""`warm` and `cold (nonce)`: the per-endpoint note `method_line` already says once."""


def method_line(document: Document) -> str:
    """The run-wide cache-mode note, once, or `""` when there is no rewritten wording for it."""
    options = as_document(document.get("options")) or {}
    warm = options.get("warm")
    return _METHOD_LINES.get(warm, "") if isinstance(warm, bool) else ""


def caveats_section(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], document: Document
) -> tuple[str, dict[int, str]]:
    """The `<h2>Caveats</h2>` list and, for a burst note, which summary index it explains.

    Each caveat starts with the short table label, bold, never the full spec string; the
    run-wide cache-mode note is skipped here because `method_line` already said it once,
    under the header, instead of once per endpoint.
    """
    items: list[str] = []
    tok_footnotes: dict[int, str] = {}
    counter = 0
    for index, (summary, label) in enumerate(zip(summaries, labels, strict=True)):
        for note in summary.notes:
            if note in _METHOD_NOTES:
                continue
            counter += 1
            anchor = f"cav-{counter}"
            if "burst" in note.lower():
                tok_footnotes[index] = anchor
            items.append(f'<li id="{anchor}"><strong>{escape(label)}</strong> {escape(note)}</li>')
    items.extend(f"<li>{escape(line)}</li>" for line in _selection_lines(document))
    if not items:
        return "", {}
    html = (
        f'<section class="caveats"><h2>Caveats</h2><ul class="notes">{"".join(items)}</ul>'
        "</section>"
    )
    return html, tok_footnotes


def _selection_lines(document: Document) -> list[str]:
    """A sweep's selection as caveat lines, or nothing for a probe whose endpoints were named.

    The report is the artifact that gets shared, so the criteria the run chose by and the
    endpoints they dropped belong in the file next to the numbers they explain.
    """
    block = as_document(document.get("sweep"))
    if block is None:
        return []
    sweep = SweepInfo.model_validate(block)
    return [selection_line(sweep), *not_probed_lines(sweep)]


def tok_footnote_cells(
    block: TableBlock, tok_footnotes: Mapping[int, str]
) -> dict[tuple[int, int], str]:
    """A footnote marker on a blank `tok/s` cell that a burst caveat explains.

    The burst note is about one rung, but the cell it explains is the per-provider row's
    median: that is the cell a reader actually looks at and finds empty, so the link goes
    there rather than to the rung table.
    """
    if "tok/s" not in block.columns or not tok_footnotes:
        return {}
    column = block.columns.index("tok/s")
    cells: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(block.rows):
        anchor = tok_footnotes.get(row_index)
        if anchor is not None and row[column] == "-":
            cells[row_index, column] = f' <sup><a href="#{anchor}">note</a></sup>'
    return cells
