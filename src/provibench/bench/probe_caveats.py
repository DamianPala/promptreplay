"""The probe page's method sentence and its Caveats section.

`ProbeSummary.notes` carries several kinds of fact at once: the run-wide cache-mode note
(`probe` puts it on every endpoint alike, since the text report prints notes per row), a
burst delivery, an output-token overspend, drift and fallback observations. The HTML page
sorts these into a fixed reading order (`caveats_section`) instead of the interleaved,
per-endpoint list the text report prints, and states the run's own shape once, above the
tables, instead of once per endpoint (`run_method_sentence`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from html import escape

from provibench.bench.labels import join_and, size_label
from provibench.bench.probe_charts import reference_size
from provibench.bench.probe_html_tables import BLANKS_TEXT
from provibench.bench.probe_summary import ProbeSummary
from provibench.bench.probe_tables import TableBlock
from provibench.bench.selection import SweepInfo, not_probed_lines, selection_line
from provibench.bench.spend import run_cost_sentence
from provibench.bench.summary import cache_mode_note, cache_mode_sentence
from provibench.core.documents import Document, as_document, as_list

__all__ = ["caveats_section", "run_method_sentence", "tok_footnote_cells"]

_NONCE_CAVEATS: dict[bool, str] = {
    True: cache_mode_sentence(True),
    False: cache_mode_sentence(False),
}
"""The run-wide cache-mode caveat, written once instead of once per endpoint. Keyed by
`options.warm`, both built from `cache_mode_sentence` so this caveat and the text report's
first note line (`probe_tables.probe_note_lines`) never say the mode in two different words."""

_METHOD_NOTES = (cache_mode_note(True), cache_mode_note(False))
"""`warm` and `cold (nonce)`: the short per-endpoint note `_nonce_caveat` already covers once."""

_OUTPUT_TOKENS_PREFIX = "ignored the one-token limit"


def run_method_sentence(summaries: Sequence[ProbeSummary], document: Document) -> str | None:
    """The sentence above the endpoint table: how many turns, at what sizes, how many repeats.

    `None` for an empty report -- there is no turn to describe.
    """
    rungs = sorted({rung.rung for summary in summaries for rung in summary.rungs})
    if not rungs:
        return None
    sizes = [size_label(size) for rung in rungs if (size := reference_size(summaries, rung)) > 0]
    if not sizes:
        return None
    options = as_document(document.get("options")) or {}
    repeats = [value for value in as_list(options.get("repeats")) or [] if isinstance(value, int)]
    turns = "1 turn" if len(rungs) == 1 else f"{len(rungs)} turns"
    return (
        f"The run replayed {turns} of a recorded coding session "
        f"({join_and(sizes)} prompt tokens) against each endpoint: one uncached request to "
        f"write the cache, then {_repeat_phrase(repeats)} to read it."
    )


def _repeat_phrase(repeats: Sequence[int]) -> str:
    """`2 to 6 repeat requests`, or the singular when every turn was read back once.

    A `--repeats 1` run is an ordinary way to use the probe, and `1 repeat requests` in the
    first sentence a reader meets reads as a template nobody finished.
    """
    if not repeats:
        return "a few repeat requests"
    low, high = min(repeats), max(repeats)
    if low != high:
        return f"{low} to {high} repeat requests"
    return "1 repeat request" if low == 1 else f"{low} repeat requests"


def caveats_section(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], document: Document
) -> tuple[str, dict[int, str]]:
    """The `<h2>Caveats</h2>` list, in the fixed order item 12 asks for, and which endpoint
    index each burst sentence explains (for the `tok/s` cell's footnote link)."""
    items: list[str] = []
    tok_footnotes: dict[int, str] = {}
    counter = 0

    def add(html: str) -> str:
        nonlocal counter
        counter += 1
        anchor = f"cav-{counter}"
        items.append(f'<li id="{anchor}">{html}</li>')
        return anchor

    nonce = _nonce_caveat(document)
    if nonce:
        add(escape(nonce))

    for line in _selection_lines(document):
        add(escape(line))

    _add_burst_caveats(add, summaries, labels, tok_footnotes)
    _add_output_token_caveats(add, summaries, labels)

    cost = run_cost_sentence(document)
    if cost:
        add(escape(cost))

    add(BLANKS_TEXT)

    _add_remaining_notes(add, summaries, labels)

    if not items:
        return "", {}
    html = (
        f'<section class="caveats"><h2>Caveats</h2><ul class="notes">{"".join(items)}</ul>'
        "</section>"
    )
    return html, tok_footnotes


def _add_burst_caveats(
    add: Callable[[str], str],
    summaries: Sequence[ProbeSummary],
    labels: Sequence[str],
    tok_footnotes: dict[int, str],
) -> None:
    """One caveat per distinct burst-turn set, and the footnote anchor each one explains."""
    for indices, sentence in _burst_sentences(summaries, labels):
        anchor = add(escape(sentence))
        for index in indices:
            tok_footnotes[index] = anchor


def _add_output_token_caveats(
    add: Callable[[str], str], summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> None:
    """Item 12.4: the output-tokens overspend note, named with the endpoint it explains."""
    for summary, label in zip(summaries, labels, strict=True):
        note = next((n for n in summary.notes if n.startswith(_OUTPUT_TOKENS_PREFIX)), None)
        if note:
            add(escape(f"{label} {note}"))


def _add_remaining_notes(
    add: Callable[[str], str], summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> None:
    """Item 12.7: every note not already given its own caveat slot, as today's per-row notes.

    The endpoint is named, not bolded: the caveats above it name their endpoints inside a
    sentence, and one bold label in a list of six plain ones reads as a different kind of
    item rather than as emphasis.
    """
    for summary, label in zip(summaries, labels, strict=True):
        for note in summary.notes:
            if _is_categorized(note):
                continue
            add(escape(f"{label} {note}"))


def _is_categorized(note: str) -> bool:
    """Whether `note` already has its own caveat slot, so it is skipped from the leftovers."""
    return (
        note in _METHOD_NOTES or "burst" in note.lower() or note.startswith(_OUTPUT_TOKENS_PREFIX)
    )


def _nonce_caveat(document: Document) -> str:
    options = as_document(document.get("options")) or {}
    warm = options.get("warm")
    return _NONCE_CAVEATS.get(warm, "") if isinstance(warm, bool) else ""


def _burst_sentences(
    summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> list[tuple[list[int], str]]:
    """One sentence per distinct set of burst turns, naming every endpoint that shares it.

    A rung bursts when its streamed request produced a time to first token but no
    generation rate (`RungSummary.gen_tok_s is None` while `ttft_ms` is not): the same
    condition `probe_stream.StreamResult.burst` records, read back from the summary alone.

    This re-derives the condition instead of reading `probe_notes._burst_note`'s own text,
    because that note does not carry the turn size this sentence needs; the two definitions
    of "burst" must keep agreeing, since `_is_categorized` suppresses the raw note by matching
    the substring `burst` in it, with no other sentence to fall back to if they diverge.
    """
    groups: dict[frozenset[int], list[int]] = {}
    for index, summary in enumerate(summaries):
        rungs = frozenset(
            rung.rung
            for rung in summary.rungs
            if rung.ttft_ms is not None and rung.gen_tok_s is None
        )
        if rungs:
            groups.setdefault(rungs, []).append(index)
    return [
        (indices, _burst_sentence(summaries, labels, indices, rungs))
        for rungs, indices in groups.items()
    ]


def _burst_sentence(
    summaries: Sequence[ProbeSummary],
    labels: Sequence[str],
    indices: Sequence[int],
    rungs: frozenset[int],
) -> str:
    names = join_and([labels[index] for index in indices])
    sizes = join_and(
        [
            size_label(size)
            for rung in sorted(rungs)
            if (size := reference_size(summaries, rung)) > 0
        ]
    )
    noun = "turn" if len(rungs) == 1 else "turns"
    pronoun = "that turn" if len(rungs) == 1 else "those turns"
    return (
        f"{names} delivered the {sizes}-token {noun}'s answer in one burst, so tok/s for "
        f"{pronoun} is -; their tok/s in the summary is the median of the other turns."
    )


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
    """A footnote marker on a `tok/s` cell reading `-` that a burst caveat explains.

    The burst note is about one turn, but the cell it explains is the per-provider row's
    median: that is the cell a reader actually looks at and finds unmeasured, so the link
    goes there rather than to the per-turn table.
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
