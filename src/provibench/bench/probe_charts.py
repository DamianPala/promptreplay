"""The probe's chart data: which endpoint stands where, and what hovering a bar says.

Kept apart from the SVG that draws it and from the page that assembles it, because this is
the part with decisions in it — how the bars are grouped, which rungs have a rate at all,
and what numbers a hover carries. The output is data (`BarGroup`/`PairedBarRow`), not markup,
so the whole thing is testable without rendering a page.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.html_svg import MAX_SERIES, Bar, BarGroup, PairedBar, PairedBarRow
from provibench.bench.labels import size_label
from provibench.bench.probe_html_tables import trace_money
from provibench.bench.probe_summary import ProbeSummary, RungSummary

_PRICE_SOURCE_WORDS: dict[str, str] = {
    "targets": "native price list",
    "openrouter-endpoint": "OpenRouter listing",
}
"""The known `price_source` values in the reader's words (item 14); anything else prints as
the tool already names it -- a fitted or a listing reprice, which are not "the" price list."""


def series(
    summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> list[tuple[ProbeSummary, str]]:
    """The endpoints that get a colour: the first `MAX_SERIES`, in document order.

    Past eight a grouped bar chart stops being readable however it is coloured — a ninth
    hue would repeat one of the eight — so the rest stay in the tables and
    `not_charted` says which they are. Both charts take the same list, so an endpoint wears
    one colour on the page.
    """
    return list(zip(summaries, labels, strict=True))[:MAX_SERIES]


def reference_size(summaries: Sequence[ProbeSummary], rung: int) -> int:
    """One rung's prompt size, from the reference endpoint when it measured it.

    Every endpoint tokenizes the same recorded turn a little differently, so the size shown
    next to a rung is always the same endpoint's, the one every drift marker is judged
    against, rather than whichever endpoint happens to be first in the document.
    """
    ranked = sorted(summaries, key=lambda summary: not summary.reference)
    return next(
        (
            candidate.prompt_cold
            for summary in ranked
            for candidate in summary.rungs
            if candidate.rung == rung and candidate.prompt_cold > 0
        ),
        0,
    )


def rung_groups(charted: Sequence[tuple[ProbeSummary, str]]) -> list[BarGroup]:
    """One group per rung, one bar per endpoint that measured it, side by side inside the group.

    Grouping by prompt size instead would split the endpoints of one rung into several
    groups as soon as they tokenize the prompt differently, and an x axis reading
    `20k 20k 20k` is no axis; the rung is what the bars are compared at. The tick names the
    reference endpoint's size for that rung, with `size_label`, so a reader still sees how
    big it was.
    """
    numbers = sorted(
        {rung.rung for summary, _ in charted for rung in summary.rungs if _measured(rung)}
    )
    return [
        BarGroup(
            label=_rung_label(number, charted),
            bars=[
                None if (rung := _at(summary, number)) is None else _bar(index, label, rung)
                for index, (summary, label) in enumerate(charted)
            ],
        )
        for number in numbers
    ]


def _rung_label(number: int, charted: Sequence[tuple[ProbeSummary, str]]) -> str:
    """`turn 2 · 41k`, the size taken from the endpoint the others are judged against."""
    size = reference_size([summary for summary, _ in charted], number)
    return f"turn {number} · {size_label(size)}" if size else f"turn {number}"


def _bar(index: int, label: str, rung: RungSummary) -> Bar:
    """One endpoint's bar at one rung: `h` (hit x prefix), the quantity the price uses.

    A rung with no cached fraction has no `h` to weight by -- it never had a hit, so `h`
    would be zero either way -- and the bar falls back to the bare hit rate with a word
    saying so, rather than silently charting a different number than the other bars.

    The hover speaks the page's words (`turn`, `repeats`, the short `@tag` label), not the
    tool's (`rung`, `warm reads`, the full spec string): it is the one text a reader meets
    without the table's tooltips next to it.
    """
    value, fell_back = _h_or_fallback(rung)
    parts = [
        label,
        f"turn {rung.rung}",
        f"{rung.prompt_cold:,} prompt tokens",
        f"{_reads(rung)} repeats hit",
    ]
    if rung.cached_fraction is not None:
        parts.append(f"cached {rung.cached_fraction * 100:.1f}%")
    if fell_back:
        parts.append("no cached share measured: bar shows the hit rate")
    return Bar(index, value, " · ".join(parts))


def _h_or_fallback(rung: RungSummary) -> tuple[float, bool]:
    """`(h, fell_back)`: `h` when the rung has a cached fraction to weight by, else the bare
    hit rate (numerically the same when there were no hits at all) with `fell_back=True`."""
    if rung.cached_fraction is None:
        return rung.hit_rate, True
    return rung.hit_rate * rung.cached_fraction, False


def _reads(rung: RungSummary) -> str:
    """`hits/served` over the rung's warm attempts, so a failed read is not counted as a miss."""
    hits = sum(1 for hit in rung.hits if hit is not None and hit > 0)
    return f"{hits}/{sum(1 for hit in rung.hits if hit is not None)}"


def _at(summary: ProbeSummary, number: int) -> RungSummary | None:
    """The endpoint's rung of this turn index, when it has a hit rate to draw."""
    return next((rung for rung in summary.rungs if rung.rung == number and _measured(rung)), None)


def _measured(rung: RungSummary) -> bool:
    """Whether a rung has a hit rate at all: a rate needs a served cold write and a warm read."""
    return not rung.skipped and any(hit is not None for hit in rung.hits)


def not_charted(summaries: Sequence[ProbeSummary], labels: Sequence[str]) -> list[str]:
    """Why an endpoint or a rung has no bar, so a gap in the chart is never read as a zero.

    Two things go unplotted: a rung without a hit rate, and every endpoint past the
    palette's eight slots — a ninth colour would have to repeat one of the eight, and two
    endpoints the eye cannot tell apart is worse than an endpoint the reader is told to find
    in the tables.
    """
    charted = list(zip(summaries, labels, strict=True))[:MAX_SERIES]
    lines = [
        f"{label} turn {rung.rung}: "
        + ("the cold request failed" if rung.skipped else "no repeat request was served")
        for summary, label in charted
        for rung in summary.rungs
        if not _measured(rung)
    ]
    rest = [label for _, label in list(zip(summaries, labels, strict=True))[MAX_SERIES:]]
    if rest:
        lines.append(
            f"{MAX_SERIES} of {len(summaries)} endpoints are charted; the rest are in the tables "
            f"above: {', '.join(rest)}"
        )
    return lines


def session_cost_rows(
    charted: Sequence[tuple[ProbeSummary, str]], trace_prompt_tokens: int | None
) -> list[PairedBarRow]:
    """One row per endpoint: the bill at the listed cache price, paired with the measured one.

    A row where either half is unknown draws no bars at all (`PairedBar.value=None`) rather
    than one bar next to an empty slot, which would read as a zero it never measured.
    """
    return [
        PairedBarRow(
            series=index,
            label=label,
            light=_listed_bar(label, summary, trace_prompt_tokens),
            dark=_measured_bar(label, summary),
        )
        for index, (summary, label) in enumerate(charted)
    ]


def _listed_session_bill(summary: ProbeSummary, trace_prompt_tokens: int | None) -> float | None:
    """What a session like the trace would bill if every repeat request hit the cache."""
    if trace_prompt_tokens is None or summary.cache_read_price is None:
        return None
    return trace_prompt_tokens / 1e6 * summary.cache_read_price


def _listed_bar(label: str, summary: ProbeSummary, trace_prompt_tokens: int | None) -> PairedBar:
    value = _listed_session_bill(summary, trace_prompt_tokens)
    text = trace_money(value)
    title = f"{label} if every repeat had hit: {text}"
    if summary.cache_read_price is not None:
        title += f" at the listed ${summary.cache_read_price:.3f}/M cache price"
    return PairedBar(value=value, value_text=text, title=title)


def _measured_bar(label: str, summary: ProbeSummary) -> PairedBar:
    """The dark bar's hover: the bill, then how it was priced, in the page's words.

    `h` is the tool's name for hit rate times cached share; the page never introduces it, so
    the hover says what the number is instead of naming it.
    """
    value = summary.session_prompt_usd
    text = trace_money(value)
    title = f"{label} at the measured hit rate: {text}"
    details: list[str] = []
    if summary.eff_per_m_prompt is not None:
        details.append(f"eff ${summary.eff_per_m_prompt:.3f}/M")
    if summary.h is not None:
        details.append(f"the cache covered {summary.h * 100:.1f}% of prompt tokens")
    details.append(f"priced from the {price_source_words(summary.price_source)}")
    return PairedBar(value=value, value_text=text, title=f"{title} ({'; '.join(details)})")


def price_source_words(source: str) -> str:
    """A `price_source` value in the reader's words, or as the tool already names it."""
    return _PRICE_SOURCE_WORDS.get(source, source)
