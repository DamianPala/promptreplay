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

from promptreplay.bench.labels import join_and, size_label
from promptreplay.bench.probe_charts import reference_size
from promptreplay.bench.probe_summary import ProbeSummary
from promptreplay.bench.probe_tables import TableBlock
from promptreplay.bench.reported_average import reported_average_source
from promptreplay.bench.selection import SweepInfo
from promptreplay.bench.selection_text import not_probed_lines, selection_line
from promptreplay.bench.spend import run_cost_sentence
from promptreplay.bench.summary import cache_mode_note, cache_mode_sentence
from promptreplay.core.documents import Document, as_document, as_list

__all__ = ["caveats_section", "run_method_sentence", "tok_footnote_cells"]

_NONCE_CAVEATS: dict[bool, str] = {
    True: cache_mode_sentence(True, labelled=False, where="on this page"),
    False: cache_mode_sentence(False, labelled=False, where="on this page"),
}
"""The run-wide cache-mode caveat's first sentence, written once instead of once per
endpoint. Keyed by `options.warm`, both built from `cache_mode_sentence` so this sentence and
the text report's first note line (`probe_tables.probe_note_lines`) never say the mode in two
different words. Unlabelled: the page shows `cold (nonce)` nowhere else, so a quoted label
here would point at nothing. The cold (nonce) mode's caveat gets a second, page-only sentence
naming where the marker sits (`_nonce_caveat`); the text report's note line stays this one
sentence alone."""

_METHOD_NOTES = (cache_mode_note(True), cache_mode_note(False))
"""`warm` and `cold (nonce)`: the short per-endpoint note `_nonce_caveat` already covers once."""

_OUTPUT_TOKENS_PREFIX = "ignored the one-token limit"

_CACHED_COLD_MARK = "reported cached tokens"
"""The per-endpoint cached-cold note (`probe_notes._cached_cold_note`), which the page folds
into one sentence for every endpoint (`_cached_cold_caveat`) instead of listing per row."""

_MARKER_POSITION = "The marker sits at the request's very first token."
"""Where the cold (nonce) mode's marker sits, in the words a provider engineer would want:
a block-aligned prefix cache can legitimately report a shared system prefix as cached when
the marker sits after it, and this run's marker does not."""

_MARKER_POSITION_WITH_CACHED_COLD = (
    "The marker sits at the request's very first token, so the next caveat's cold writes "
    "cannot be explained by a shared prefix."
)
"""Said instead of `_MARKER_POSITION` when the cached-cold caveat also runs: the two caveats
would otherwise sit side by side making contradictory-looking claims, one nonce hit is always
new and the other says a cold write already read one back, with nothing on the page saying
why both are true."""


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
    sentence = (
        f"The run replayed {turns} of a recorded coding session "
        f"({join_and(sizes)} prompt tokens) against each endpoint: one uncached request to "
        f"write the cache, then {_repeat_phrase(repeats)} to read it."
    )
    day = _reported_day(summaries, document)
    if day is not None:
        sentence += (
            " The section below sets this run's cache share against OpenRouter's reported "
            f"average for {day}."
        )
    return sentence


def _reported_day(summaries: Sequence[ProbeSummary], document: Document) -> str | None:
    """The run's reported-average day, but only while a summary carries a figure to show.

    A fetch can come back with no share for any endpoint the run probed (a provider absent
    from that day's row, a price the run never learned); `reported_average_section` then
    renders nothing, so neither the method sentence nor the source caveat may point at it.
    """
    if not any(summary.or_avg_share_pct is not None for summary in summaries):
        return None
    reported = as_document(document.get("reported_average"))
    day = reported.get("day") if reported is not None else None
    return day if isinstance(day, str) else None


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

    cached_cold = _cached_cold_caveat(summaries, labels, document)
    nonce = _nonce_caveat(document, cached_cold_present=cached_cold is not None)
    if nonce:
        add(escape(nonce))
    if cached_cold:
        add(escape(cached_cold))
    source = _reported_average_caveat(summaries, document)
    if source:
        add(escape(source))

    for line in _selection_lines(document):
        add(escape(line))

    _add_burst_caveats(add, summaries, labels, tok_footnotes)
    _add_output_token_caveats(add, summaries, labels)

    cost = run_cost_sentence(document)
    if cost:
        add(escape(cost))

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
        note in _METHOD_NOTES
        or "burst" in note.lower()
        or note.startswith(_OUTPUT_TOKENS_PREFIX)
        or _CACHED_COLD_MARK in note
    )


def _cached_cold_caveat(
    summaries: Sequence[ProbeSummary], labels: Sequence[str], document: Document
) -> str | None:
    """One sentence for every endpoint whose cold writes reported cached tokens.

    The text report says it once per endpoint (`probe_notes`), which is its per-row shape;
    six near-identical lines on the page bury the one fact they share, so the page names
    the endpoints inside one sentence and gives each its count. Nothing under `--warm`,
    where a cold write hitting the cache is the thing the run set out to measure.
    """
    options = as_document(document.get("options")) or {}
    if options.get("warm") is True:
        return None
    parts: list[str] = []
    for summary, label in zip(summaries, labels, strict=True):
        colds = [rung for rung in summary.rungs if rung.prompt_cold > 0]
        hit = sum(1 for rung in colds if rung.cached_cold > 0)
        if hit:
            noun = "turn" if len(colds) == 1 else "turns"
            parts.append(f"{label} on {hit} of {len(colds)} {noun}")
    if not parts:
        return None
    plural = len(parts) > 1
    verdict = (
        "Their caches are not strict prefix caches, or their counts are not what they say."
        if plural
        else "Its cache is not a strict prefix cache, or its count is not what it says."
    )
    return (
        f"Cold writes reported cached tokens although the nonce made them new prompts: "
        f"{join_and(parts)}. {verdict}"
    )


def _reported_average_caveat(summaries: Sequence[ProbeSummary], document: Document) -> str | None:
    """Where the OR avg figures come from, right after the cached-cold caveat."""
    day = _reported_day(summaries, document)
    return reported_average_source(day) if day is not None else None


def _nonce_caveat(document: Document, *, cached_cold_present: bool) -> str:
    """The run-wide cache-mode caveat: `_NONCE_CAVEATS`'s shared sentence, plus, for the cold
    (nonce) mode only, where the marker sits. `cached_cold_present` picks which of the two
    marker sentences applies, since only then does the position need to also explain away
    the objection that a shared prefix could account for the next caveat's cold writes.
    """
    options = as_document(document.get("options")) or {}
    warm = options.get("warm")
    if not isinstance(warm, bool):
        return ""
    base = _NONCE_CAVEATS[warm]
    if warm:
        return base
    marker = _MARKER_POSITION_WITH_CACHED_COLD if cached_cold_present else _MARKER_POSITION
    return f"{base} {marker}"


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
    pronoun = "that turn has" if len(rungs) == 1 else "those turns have"
    others = len({rung.rung for summary in summaries for rung in summary.rungs}) - len(rungs)
    return (
        f"{names} delivered the {sizes}-token {noun}'s answer in one burst, so {pronoun} no "
        f"tok/s (shown as - in the per-turn table). {_summary_tok_phrase(others)}."
    )


def _summary_tok_phrase(others: int) -> str:
    """What the summary's `tok/s` cell holds once the burst turns are left out of it.

    The only `-` cells a reader meets are the per-turn ones this sentence explains, so the
    explanation lives here rather than in a caveat of its own about the symbol. Capitalized,
    because it follows the burst sentence as a sentence of its own.
    """
    if others <= 0:
        return "Their tok/s in the summary is - as well"
    if others == 1:
        return "Their tok/s in the summary is the other turn's"
    return f"Their tok/s in the summary is the median of the other {_count_word(others)} turns"


_COUNT_WORDS: dict[int, str] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _count_word(count: int) -> str:
    return _COUNT_WORDS.get(count, str(count))


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

    The burst note is about one turn, but the cell it explains is the per-endpoint row's
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
