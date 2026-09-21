"""Applying OpenRouter's reported average to a probe's summaries, and the words for it.

`bench.openrouter_stats` fetches one day's reported cache share per provider; this module
answers "so what": which summary it applies to, what it changes the eff price and the
session bill into, and the sentences and table a reader meets on the page. Kept apart from
`openrouter_stats` because that module is the network boundary and this one is pure
arithmetic and templates, the same split `probe_drift` keeps from `openrouter`.

Only a pinned OpenRouter spec is in scope: the reported share is per provider, so an
unpinned spec (which may have been served by any of them) and a native spec (which OpenRouter
never sees) get none of the five fields this module fills.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from promptreplay.bench.labels import join_and
from promptreplay.bench.openrouter_stats import ReportedAverage
from promptreplay.bench.pricing import effective_price
from promptreplay.bench.probe_drift import SpecRef
from promptreplay.bench.probe_html_tables import trace_money
from promptreplay.bench.probe_summary import ProbeSummary
from promptreplay.bench.probe_tables import TableBlock
from promptreplay.bench.targets import Prices

_NUMBER_WORDS: dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

_BLOCK_COLUMNS = (
    "endpoint",
    "cache %",
    "OR avg",
    "eff $/M",
    "OR avg",
    "this trace $",
    "OR avg",
    "vs OR avg",
)


def apply_reported_average(
    summaries: Sequence[ProbeSummary],
    specs: Sequence[SpecRef],
    average: ReportedAverage | None,
    *,
    trace_prompt_tokens: int | None,
) -> list[ProbeSummary]:
    """Fill the five `or_avg_*`/`vs_or_avg_pct` fields for every spec the feed covers.

    A native spec, an unpinned OpenRouter spec, a pinned one whose provider the feed did not
    report that day, or one naming a different model than the fetch was made for, comes back
    as itself -- like `probe_drift.apply_drift`, only the specs the comparison applies to are
    copied.
    """
    if average is None:
        return list(summaries)
    return [
        _apply_one(summary, spec, average, trace_prompt_tokens=trace_prompt_tokens)
        for summary, spec in zip(summaries, specs, strict=True)
    ]


def _apply_one(
    summary: ProbeSummary,
    spec: SpecRef,
    average: ReportedAverage,
    *,
    trace_prompt_tokens: int | None,
) -> ProbeSummary:
    if spec.kind != "openrouter" or not spec.providers:
        return summary
    if not average.model or spec.model != average.model:
        # A block with no model at all (written before the field existed) names no owner for
        # its shares, so it applies to no spec rather than to every spec of that provider.
        return summary
    input_price = summary.input_price
    cache_read_price = summary.cache_read_price
    if input_price is None or cache_read_price is None:
        return summary
    slug = spec.providers[0].partition("/")[0]
    share = average.shares.get(slug)
    if share is None:
        return summary
    or_avg_eff = _or_avg_effective_price(input_price, cache_read_price, share.share_pct)
    or_avg_session = (
        or_avg_eff * trace_prompt_tokens / 1e6 if trace_prompt_tokens is not None else None
    )
    return summary.model_copy(
        update={
            "or_avg_share_pct": share.share_pct,
            "or_avg_pooled": share.endpoints > 1,
            "or_avg_eff_per_m_prompt": or_avg_eff,
            "or_avg_session_prompt_usd": or_avg_session,
            "vs_or_avg_pct": _vs_or_avg_pct(
                summary, or_avg_eff, or_avg_session, trace_prompt_tokens
            ),
        }
    )


def _or_avg_effective_price(input_price: float, cache_read_price: float, share_pct: float) -> float:
    """The same `eff $/M` arithmetic (`bench.pricing.effective_price`), at `share_pct` instead
    of this run's own share, and at this run's own listed prices."""
    prices = Prices(input=input_price, cache_read=cache_read_price, cache_write=0.0, output=0.0)
    # `effective_price` returns `None` only when either argument is `None`; neither is here.
    return cast("float", effective_price(share_pct / 100, prices))


def _vs_or_avg_pct(
    summary: ProbeSummary,
    or_avg_eff: float,
    or_avg_session: float | None,
    trace_prompt_tokens: int | None,
) -> float | None:
    """This run's bill against OpenRouter's average bill; the eff-price ratio when the trace
    size (and so the session bill) is unknown, since the two ratios are the same number."""
    if (
        trace_prompt_tokens is not None
        and or_avg_session is not None
        and summary.session_prompt_usd is not None
        and or_avg_session != 0
    ):
        return (summary.session_prompt_usd / or_avg_session - 1) * 100
    if summary.eff_per_m_prompt is not None and or_avg_eff != 0:
        return (summary.eff_per_m_prompt / or_avg_eff - 1) * 100
    return None


def _count_word(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


def _trace_words(tokens: int) -> str:
    """A trace's total prompt tokens as the caption states it: `1.8M`, not `1804855`."""
    if tokens >= 1_000_000:
        return f"{tokens / 1e6:.1f}M"
    return f"{round(tokens / 1000)}k"


def reported_average_caption(day: str, trace_prompt_tokens: int | None) -> str:
    """The HTML page's caption above the comparison table: what it shows, and how to read a
    negative `vs OR avg`."""
    subject = "The recorded session's input tokens are priced"
    if trace_prompt_tokens is not None and trace_prompt_tokens > 0:
        subject = (
            f"The recorded session's {_trace_words(trace_prompt_tokens)} input tokens are priced"
        )
    return (
        "OpenRouter's model page reports, per endpoint and day, the share of its customers' "
        f"prompt tokens served from cache. {subject} twice at the same listed prices: at this "
        f"run's share and at OpenRouter's reported share for {day}. A negative `vs OR avg` "
        "means this session cached better on that endpoint than OpenRouter's average traffic "
        "did."
    )


def reported_average_table_caption(day: str) -> str:
    """The one-line caption the text and Markdown tables print above the comparison table."""
    return (
        f"vs OpenRouter's reported average for {day} (same listed prices, its cache share in "
        "place of this run's):"
    )


def reported_average_sentence(
    summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> str | None:
    """How many endpoints cached better this session than OpenRouter's average traffic did.

    `None` when no summary carries a figure at all. The "below" clause is left out when
    nothing was below, and the "above" count is left out when nothing was above, so a
    one-endpoint run never reads "on 0 of the one endpoints".
    """
    pairs = [
        (label, summary.or_avg_share_pct, (summary.h or 0.0) * 100)
        for summary, label in zip(summaries, labels, strict=True)
        if summary.or_avg_share_pct is not None and summary.h is not None
    ]
    if not pairs:
        return None
    above = sum(1 for _, share, run in pairs if run > share)
    below = [label for label, share, run in pairs if run < share]
    if not above and not below:
        return f"This session's share matches OpenRouter's average on {_endpoints(len(pairs))}."
    if not above:
        return f"This session's share sits below OpenRouter's average on {join_and(below)}."
    where = (
        "the one endpoint"
        if len(pairs) == 1
        else f"{_count_word(above)} of the {_count_word(len(pairs))} endpoints"
    )
    sentence = f"This session's share sits above OpenRouter's average on {where}"
    if below:
        sentence += f" and below it on {join_and(below)}"
    return sentence + "."


def _endpoints(count: int) -> str:
    """`the one endpoint` or `all six endpoints`."""
    return "the one endpoint" if count == 1 else f"all {_count_word(count)} endpoints"


def reported_average_source(day: str) -> str:
    """The caveat naming where the OR avg figures come from, and their exact scope for `day`."""
    return (
        "The OR avg figures come from the feed behind OpenRouter's model page, not from its "
        f"documented API, and may change without notice. Each is one day's total ({day}) over "
        "every OpenRouter customer of that endpoint, routed or pinned, of the prompt tokens "
        "the provider reported as cached."
    )


def pooled_note() -> str:
    """The footnote for a `*`-marked cell: a pooled figure covers more than one endpoint."""
    return (
        "* The feed names endpoints by provider only, so rows of one provider carry its "
        "token-weighted share."
    )


def reported_average_block(
    summaries: Sequence[ProbeSummary], labels: Sequence[str]
) -> TableBlock | None:
    """The comparison table: one row per summary the feed covered, in the summaries' order."""
    rows = [
        reported_average_row(summary, label)
        for summary, label in zip(summaries, labels, strict=True)
        if summary.or_avg_share_pct is not None
    ]
    if not rows:
        return None
    return TableBlock(columns=_BLOCK_COLUMNS, rows=tuple(rows))


def reported_average_row(summary: ProbeSummary, label: str) -> tuple[str, ...]:
    """One row's cells, in `_BLOCK_COLUMNS` order -- also `html_reported`'s source for the
    same numbers, so a row can never disagree between the text/Markdown and HTML renderers."""
    star = "*" if summary.or_avg_pooled else ""
    run_share = summary.h * 100 if summary.h is not None else None
    return (
        label,
        _pct1(run_share),
        f"{_pct1(summary.or_avg_share_pct)}{star}",
        _money3(summary.eff_per_m_prompt),
        f"{_money3(summary.or_avg_eff_per_m_prompt)}{star}",
        trace_money(summary.session_prompt_usd),
        trace_money(summary.or_avg_session_prompt_usd),
        _vs_cell(summary.vs_or_avg_pct),
    )


def _pct1(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _money3(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def _vs_cell(pct: float | None) -> str:
    if pct is None:
        return "-"
    if abs(pct) < 0.5:
        return "0%"
    return f"{pct:+.0f}%"
