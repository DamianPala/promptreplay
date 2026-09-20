"""The report's one-line answer: which endpoint is cheapest, and by how much.

This is the highest-value change the report's own user review asked for: a reader who has
not opened the README wants two things in a minute -- which endpoint to use, and what the
difference costs -- and every number the sentence prints is already on the page. This module
only decides how to say it, from the same `ProbeSummary` list the tables and charts read.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.probe_summary import ProbeSummary

__all__ = ["answer_line", "join_and", "trace_money"]


def answer_line(summaries: Sequence[ProbeSummary], labels: Sequence[str]) -> str | None:
    """One sentence naming the cheapest endpoint(s) and, when there is one, the priciest.

    `None` for an empty report. A tie keeps every endpoint at the winning price in one
    group; an endpoint with no effective price is left out of the comparison rather than
    guessed at, and when none of them have one the sentence says so instead of a number.
    """
    if not summaries:
        return None
    priced: list[tuple[ProbeSummary, str, float]] = [
        (summary, label, price)
        for summary, label in zip(summaries, labels, strict=True)
        if (price := summary.eff_per_m_prompt) is not None
    ]
    if not priced:
        if all(summary.errors > 0 for summary in summaries):
            return "Every endpoint errored on this trace, so there is no price to compare."
        return "No endpoint's effective price could be measured on this trace."
    cheapest_price = min(price for _, _, price in priced)
    cheapest = [entry for entry in priced if _close(entry[2], cheapest_price)]
    others = [entry for entry in priced if not _close(entry[2], cheapest_price)]
    names, verb = _group_phrase([label for _, label, _ in cheapest])
    core = f"{names} {verb} ${cheapest_price:.3f} per 1M prompt tokens on this trace"
    cheap_summary = cheapest[0][0]
    if not others:
        return core + _session_clause(cheap_summary, cheap_summary) + "."
    priciest_price = max(price for _, _, price in others)
    priciest = [entry for entry in others if _close(entry[2], priciest_price)]
    priciest_names, priciest_verb = _group_phrase([label for _, label, _ in priciest])
    multiple = priciest_price / cheapest_price if cheapest_price > 0 else None
    tail = f", about {_multiple_text(multiple)} more" if multiple is not None else ""
    core += f"; {priciest_names} {priciest_verb} ${priciest_price:.3f}{tail}"
    return core + _session_clause(cheap_summary, priciest[0][0]) + "."


def join_and(names: Sequence[str]) -> str:
    """`A`, `A and B`, or `A, B and C`: a plain list, no Oxford comma past two items."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def trace_money(value: float | None) -> str:
    """A trace-shaped dollar amount as the reader thinks in it: `$0.0060`, not `$/M`.

    Four decimals, always: this is the same `session_prompt_usd` the text report already
    prints through `commands.probe_spend`, so the two renderers would otherwise disagree
    over one number. A fixed count is also what a column is read down -- trimming zeros put
    `$0.006` and `$0.0056` in neighbouring rows, which has to be compared digit by digit --
    and at three decimals a cheap session rounds to `$0.000`, which reads as free.
    """
    return "-" if value is None else f"${value:.4f}"


def _close(a: float, b: float) -> bool:
    """Same price at the three decimals the table and this sentence both print it to.

    Two independently hit-weighted prices rarely land on the exact same float even when a
    reader would call them tied -- the table already rounds to three decimals, so this
    sentence groups by the same rounding rather than by an exact-equality epsilon that
    would silently drop a tied endpoint out of both the cheapest and the priciest group.
    """
    return round(a, 3) == round(b, 3)


def _group_phrase(names: Sequence[str]) -> tuple[str, str]:
    if len(names) == 1:
        return names[0], "costs"
    if len(names) == 2:
        return f"{names[0]} and {names[1]}", "both cost"
    return join_and(names), "all cost"


def _multiple_text(multiple: float) -> str:
    rounded = round(multiple)
    return f"{multiple:.1f}x" if rounded <= 1 else f"{rounded}x"


def _session_clause(cheap: ProbeSummary, pricey: ProbeSummary) -> str:
    """`, for a session like this trace that is $X[, vs $Y]`, or nothing without the data."""
    if cheap.session_prompt_usd is None or pricey.session_prompt_usd is None:
        return ""
    if cheap is pricey:
        return f", for a session like this trace that is {trace_money(cheap.session_prompt_usd)}"
    return (
        ", for a session like this trace that is "
        f"{trace_money(cheap.session_prompt_usd)} vs {trace_money(pricey.session_prompt_usd)}"
    )
