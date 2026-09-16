"""What a probe run spent, folded into the document `probe` and `report` both publish.

Kept apart from `commands.probe` purely for its line budget: the numbers here are a few
lines of arithmetic over what `bench.spend` and the availability check already computed, and
moving them out keeps `execute_probe` itself the single flow it is documented as.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from provibench.core.documents import (
    Document,
    JsonSchema,
    array,
    as_document,
    integer,
    nullable_number,
    nullable_object,
    number,
    obj,
    string,
)

if TYPE_CHECKING:
    from provibench.bench.estimate import SpecPrices
    from provibench.bench.probe_models import ProbeResult
    from provibench.bench.probe_summary import ProbeSummary
    from provibench.bench.targets import Prices

PRECHECK_PROPERTIES: dict[str, JsonSchema] = {
    "requests": integer(),
    "spend_usd": nullable_number(),
    "worst_case_usd": nullable_number(),
}
PRECHECK_SCHEMA = nullable_object(PRECHECK_PROPERTIES, required=list(PRECHECK_PROPERTIES))

_LISTING_PRICE_PROPERTIES: dict[str, JsonSchema] = {
    "provider": string(),
    "input": number(),
    "cache_read": number(),
    "cache_write": number(),
    "output": number(),
}
LISTING_PRICES_SCHEMA = array(
    obj(_LISTING_PRICE_PROPERTIES, required=list(_LISTING_PRICE_PROPERTIES))
)

_ONE_MILLION = 1_000_000


def _dash_money(value: object) -> str:
    """An amount as `$0.1234` for a rendered document field, or `-` when it is unknown."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return f"${value:.4f}"


def spend_lines(document: Document) -> list[str]:
    """`pre-check: N requests, $X` then `spent $X (worst case $Y)`, when the run knows them.

    Both a `probe` result and a `report` of one carry the same `spend_usd`/`precheck` fields,
    so this is the one place either renders them.
    """
    lines: list[str] = []
    precheck = as_document(document.get("precheck"))
    if precheck is not None:
        requests = precheck.get("requests")
        lines.append(f"pre-check: {requests} requests, {_dash_money(precheck.get('spend_usd'))}")
    spend = document.get("spend_usd")
    if isinstance(spend, int | float):
        worst = _dash_money(document.get("worst_case_usd"))
        lines.append(f"spent {_dash_money(spend)} (worst case {worst})")
    return lines


def session_footer_lines(
    summaries: Sequence[ProbeSummary], trace_prompt_tokens: object
) -> list[str]:
    """The session-projection footer: what a session shaped like the trace would bill.

    One line, prompt tokens only -- `session_prompt_usd` is `eff_per_m_prompt` times the
    trace's own prompt-token total, so a spec with neither known contributes nothing rather
    than an unpriced entry the reader has to discount.
    """
    if not isinstance(trace_prompt_tokens, int) or trace_prompt_tokens <= 0:
        return []
    parts = [
        f"{summary.label} {_dash_money(summary.session_prompt_usd)}"
        for summary in summaries
        if summary.session_prompt_usd is not None
    ]
    if not parts:
        return []
    header = f"prompt bill for a session like this trace ({_token_count(trace_prompt_tokens)})"
    return [f"{header}: {', '.join(parts)}"]


def _token_count(tokens: int) -> str:
    """`N k prompt tokens` under a million, `N.N M prompt tokens` at or above it.

    A small trace (a handful of turns, well under 1 M prompt tokens) read as `0.0 M` before
    this; thousands are the unit a reader expects at that size.
    """
    if tokens < _ONE_MILLION:
        return f"{tokens / 1e3:.0f} k prompt tokens"
    return f"{tokens / 1e6:.1f} M prompt tokens"


def listing_prices_document(prices: Mapping[str, Prices] | None) -> list[Document]:
    """`listing_prices` as `[{provider, input, cache_read, cache_write, output}, ...]`.

    Sorted by provider so the document is stable across a rebuild; `None` (a run written
    before the field existed) and `{}` (a run with no OpenRouter spec to list) both render
    as an empty list -- the distinction only matters to `choose_prices`, not to a reader.
    """
    if not prices:
        return []
    return [
        {
            "provider": provider,
            "input": entry.input,
            "cache_read": entry.cache_read,
            "cache_write": entry.cache_write,
            "output": entry.output,
        }
        for provider, entry in sorted(prices.items())
    ]


def report_spend(
    summaries: Sequence[ProbeSummary],
    precheck: Sequence[ProbeResult],
    *,
    prices: Mapping[str, SpecPrices],
    records: Mapping[str, Sequence[ProbeResult]],
) -> tuple[Document | None, float | None, float | None]:
    """Fold a finished run's spend into the fields `probe` and `report` both publish.

    Returns `(precheck_document, total_spend, worst_case_usd)`. Nothing is printed here: the
    closing `spent $X (worst case $Y)` line is rendered exactly once, as the last line of the
    tables (`spend_lines`, called from `probe_report_text` and `report_markdown`), so a human
    run never sees it twice and a machine-readable one carries it only in the document.
    """
    from provibench.bench.spend import precheck_spend, precheck_worst_case, total_spend
    from provibench.bench.spend import run_worst_case_usd as _run_worst_case_usd

    precheck_spend_usd = precheck_spend(precheck, prices) if precheck else None
    precheck_worst_usd = precheck_worst_case(precheck, prices) if precheck else None
    spend = total_spend([summary.spend_usd for summary in summaries])
    total = total_spend([spend, precheck_spend_usd])
    worst = _run_worst_case_usd(records, prices, precheck_worst_case_usd=precheck_worst_usd)
    precheck_document: Document | None = (
        {
            "requests": len(precheck),
            "spend_usd": precheck_spend_usd,
            # `report` publishes this block straight from the persisted `PrecheckSummary`,
            # so `probe` carries the same three fields rather than a narrower shape under
            # the one schema both commands declare.
            "worst_case_usd": precheck_worst_usd,
        }
        if precheck
        else None
    )
    return precheck_document, total, worst
