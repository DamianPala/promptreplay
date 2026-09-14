"""The probe's chart data: which spec stands where, and what hovering a bar says.

Kept apart from the SVG that draws it and from the page that assembles it, because this is
the part with decisions in it — how the bars are grouped, which rungs have a rate at all,
and what numbers a hover carries. The output is data (`BarGroup`/`BarRow`), not markup, so
the whole thing is testable without rendering a page.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.html_svg import MAX_SERIES, Bar, BarGroup, BarRow
from provibench.bench.probe_summary import ProbeSummary, RungSummary


def series(
    summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> list[tuple[ProbeSummary, str]]:
    """The specs that get a colour: the first `MAX_SERIES`, in document order.

    Past eight a grouped bar chart stops being readable however it is coloured — a ninth
    hue would repeat one of the eight — so the rest stay in the tables and
    `not_charted` says which they are. Both charts take the same list, so a spec wears one
    colour on the page.
    """
    return list(zip(summaries, labels, strict=True))[:MAX_SERIES]


def _size_label(tokens: int) -> str:
    """A rung's prompt size as the x tick shows it: `20k`, or the token count below 1k."""
    return f"{tokens / 1000:.0f}k" if tokens >= 1000 else str(tokens)


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def rung_groups(charted: Sequence[tuple[ProbeSummary, str]]) -> list[BarGroup]:
    """One group per rung, one bar per spec that measured it, side by side inside the group.

    Grouping by prompt size instead would split the specs of one rung into several groups
    as soon as they tokenize the prompt differently, and an x axis reading `20k 20k 20k` is
    no axis; the rung is what the bars are compared at. The tick names the reference spec's
    size for that rung, with `_size_label`, so a reader still sees how big it was.
    """
    numbers = sorted(
        {rung.rung for summary, _ in charted for rung in summary.rungs if _measured(rung)}
    )
    return [
        BarGroup(
            label=_rung_label(number, charted),
            bars=[
                None if (rung := _at(summary, number)) is None else _bar(index, summary, rung)
                for index, (summary, _) in enumerate(charted)
            ],
        )
        for number in numbers
    ]


def _rung_label(number: int, charted: Sequence[tuple[ProbeSummary, str]]) -> str:
    """`rung 2 · 41k`, the size taken from the spec the others are judged against."""
    ranked = sorted(charted, key=lambda pair: not pair[0].reference)
    size = next(
        (
            rung.prompt_cold
            for summary, _ in ranked
            for rung in summary.rungs
            if rung.rung == number and rung.prompt_cold > 0
        ),
        0,
    )
    return f"rung {number} · {_size_label(size)}" if size else f"rung {number}"


def _bar(index: int, summary: ProbeSummary, rung: RungSummary) -> Bar:
    """One spec's bar at one rung, with the numbers a reader would otherwise estimate."""
    parts = [
        summary.label,
        f"rung {rung.rung}",
        f"{rung.prompt_cold:,} prompt tokens",
        f"hit {rung.hit_rate * 100:.1f}% ({_reads(rung)} warm reads)",
    ]
    if rung.prefix_fraction is not None:
        parts.append(f"prefix {rung.prefix_fraction * 100:.1f}%")
    return Bar(index, rung.hit_rate, " · ".join(parts))


def _reads(rung: RungSummary) -> str:
    """`hits/served` over the rung's warm attempts, so a failed read is not counted as a miss."""
    hits = sum(1 for hit in rung.hits if hit is not None and hit > 0)
    return f"{hits}/{sum(1 for hit in rung.hits if hit is not None)}"


def _at(summary: ProbeSummary, number: int) -> RungSummary | None:
    """The spec's rung of this turn index, when it has a hit rate to draw."""
    return next((rung for rung in summary.rungs if rung.rung == number and _measured(rung)), None)


def _measured(rung: RungSummary) -> bool:
    """Whether a rung has a hit rate at all: a rate needs a served cold write and a warm read."""
    return not rung.skipped and any(hit is not None for hit in rung.hits)


def not_charted(summaries: Sequence[ProbeSummary], labels: Sequence[str]) -> list[str]:
    """Why a spec or a rung has no bar, so a gap in the chart is never read as a zero.

    Two things go unplotted: a rung without a hit rate, and every spec past the palette's
    eight slots — a ninth colour would have to repeat one of the eight, and two specs the
    eye cannot tell apart is worse than a spec the reader is told to find in the tables.
    """
    charted = list(zip(summaries, labels, strict=True))[:MAX_SERIES]
    lines = [
        f"{summary.label} rung {rung.rung}: "
        + ("the cold request failed" if rung.skipped else "no warm read was served")
        for summary, _ in charted
        for rung in summary.rungs
        if not _measured(rung)
    ]
    rest = [label for _, label in list(zip(summaries, labels, strict=True))[MAX_SERIES:]]
    if rest:
        lines.append(
            f"{MAX_SERIES} of {len(summaries)} specs are charted; the rest are in the tables "
            f"above: {', '.join(rest)}"
        )
    return lines


def cost_rows(charted: Sequence[tuple[ProbeSummary, str]]) -> list[BarRow]:
    """One row per spec: its effective prompt price, or `None` when no price is known."""
    return [
        BarRow(
            series=index,
            label=label,
            value=summary.eff_per_m_prompt,
            value_text=_rate(summary.eff_per_m_prompt),
            title=_cost_title(summary),
        )
        for index, (summary, label) in enumerate(charted)
    ]


def _cost_title(summary: ProbeSummary) -> str:
    """The exact numbers behind one effective-price bar."""
    parts = [summary.label, f"eff ${_rate(summary.eff_per_m_prompt)}/M prompt"]
    if summary.input_price is not None:
        parts.append(f"listed ${summary.input_price:.2f}/M in")
    if summary.cache_read_price is not None:
        parts.append(f"${summary.cache_read_price:.2f}/M cache read")
    if summary.h is not None:
        parts.append(f"hit-weighted h {summary.h * 100:.1f}%")
    parts.append(f"prices: {summary.price_source}")
    return " · ".join(parts)
