"""What a probe run actually spent: OpenRouter's own bill, or the usage priced by hand.

OpenRouter reports what it billed for every request it enriched (`ProbeResult.generation`),
so a gateway spec's spend is that number, exactly. A native spec never gets a generation
lookup, so its spend is computed the only other way it can be: the tokens it used, priced at
the run's own listed rates. Both paths price every served request the same way -- uncached
prompt tokens at the input rate, cached ones at the cache-read rate, written ones at the
cache-write rate, and the output tokens a read was not supposed to produce (see
`bench.probe_pricing`) at the output rate -- so a spec that mixes billed and estimated turns
would still sum to one honest number; in practice a spec is one or the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from promptreplay.bench.estimate import SpecPrices
from promptreplay.bench.probe_models import ProbeResult, is_served
from promptreplay.core.documents import Document, as_document

__all__ = [
    "dash_money",
    "precheck_spend",
    "precheck_worst_case",
    "record_cost",
    "record_worst_case",
    "run_cost_sentence",
    "run_worst_case_usd",
    "spec_spend",
    "total_spend",
    "usage_cost",
    "worst_case_usd",
]


def record_cost(record: ProbeResult, prices: SpecPrices | None) -> float | None:
    """One request's cost at `prices`' listed rates, or `None` without a listed price.

    `prompt_total` is Anthropic's sum of all three prompt buckets, so the tokens billed at
    the plain input rate are the ones neither read from the cache nor written to it:
    subtracting only `cached` would charge every written token twice, once here and once at
    the cache-write rate below (`bench.enrich` splits the same three buckets the same way).
    """
    if prices is None:
        return None
    uncached = max(record.prompt_total - record.cached - record.cache_write, 0)
    rates = prices.prices
    usd = (
        uncached * rates.input
        + record.cached * rates.cache_read
        + record.cache_write * rates.cache_write
        + record.usage.output_tokens * rates.output
    )
    return usd * 1e-6


def usage_cost(records: Sequence[ProbeResult], prices: SpecPrices | None) -> float | None:
    """The summed cost of every served record at `prices`, or `None` without one."""
    if prices is None:
        return None
    served = [record for record in records if is_served(record)]
    if not served:
        return None
    return sum(record_cost(record, prices) or 0.0 for record in served)


def spec_spend(
    records: Sequence[ProbeResult], *, billed_usd: float | None, prices: SpecPrices | None
) -> float | None:
    """What a spec's requests cost: what OpenRouter billed, else what the usage implies."""
    if billed_usd is not None:
        return billed_usd
    return usage_cost(records, prices)


def precheck_spend(
    records: Sequence[ProbeResult], prices: Mapping[str, SpecPrices]
) -> float | None:
    """The availability check's own spend: each candidate priced at its own listed rate.

    A pre-check request never gets an OpenRouter generation lookup, so this is always the
    usage-priced number, one candidate at a time -- `record_cost` already handles a candidate
    with no listed price by leaving its own share `None`.
    """
    served = [record for record in records if is_served(record)]
    amounts = [record_cost(record, prices.get(record.spec_label)) for record in served]
    return total_spend(amounts)


def record_worst_case(record: ProbeResult, prices: SpecPrices | None) -> float | None:
    """What one served request would have cost with every prompt token billed at `input`."""
    if prices is None:
        return None
    return record.prompt_total * prices.prices.input * 1e-6


def worst_case_usd(records: Sequence[ProbeResult], prices: SpecPrices | None) -> float | None:
    """What a spec would have cost with no cache hit: every served prompt token at `input`.

    Mirrors the estimate's own worst case (`bench.estimate`), computed after the fact from a
    persisted run's own records instead of from the trace -- so `report` can print the same
    figure years later, offline, next to what the run actually spent.
    """
    if prices is None:
        return None
    served = [record for record in records if is_served(record)]
    if not served:
        return None
    return sum(record_worst_case(record, prices) or 0.0 for record in served)


def precheck_worst_case(
    records: Sequence[ProbeResult], prices: Mapping[str, SpecPrices]
) -> float | None:
    """The availability check's own worst case: each candidate's requests at its own `input`."""
    served = [record for record in records if is_served(record)]
    amounts = [record_worst_case(record, prices.get(record.spec_label)) for record in served]
    return total_spend(amounts)


def run_worst_case_usd(
    records: Mapping[str, Sequence[ProbeResult]],
    prices: Mapping[str, SpecPrices],
    *,
    precheck_worst_case_usd: float | None = None,
) -> float | None:
    """The whole run's worst case: every served spec record priced at its own input rate with
    no cache hit, plus the pre-check's own worst case when the run planned one.

    `probe` calls this right after the run, with `precheck_worst_case_usd` fresh from
    `precheck_worst_case` over the requests it just sent; `report` calls it again with the
    same number read back from `ProbeRunMeta.precheck.worst_case_usd` -- one function
    aggregating the same per-record worst cases either way, so the two print the same total.
    """
    per_spec = [worst_case_usd(rs, prices.get(label)) for label, rs in records.items()]
    return total_spend([*per_spec, precheck_worst_case_usd])


def total_spend(amounts: Sequence[float | None]) -> float | None:
    """The sum of every known amount, or `None` when none of them are known.

    Mirrors `bench.estimate.estimate_total`: a spec or the pre-check with no listed price
    leaves its own share unknown rather than zero, but the run's total is still the sum of
    what is known, not `None` just because one part is not.
    """
    known = [amount for amount in amounts if amount is not None]
    return sum(known) if known else None


def dash_money(value: object) -> str:
    """An amount as `$0.1234` for a rendered document field, or `-` when it is unknown."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return f"${value:.4f}"


def run_cost_sentence(document: Document) -> str | None:
    """Item 12.5's caveat: one sentence from the same fields `commands.probe_spend.spend_lines`
    reads, for the HTML page's caveats section (the text report keeps `spend_lines`'s own
    closing lines). Lives here, not next to `spend_lines`, because `probe_caveats.py` (in
    `bench/`) needs it and `bench/` must not import from `commands/`.
    """
    spend = document.get("spend_usd")
    if not isinstance(spend, int | float):
        return None
    precheck = as_document(document.get("precheck"))
    precheck_spend = precheck.get("spend_usd") if precheck else None
    sentence = f"This run cost {dash_money(spend)} in API spend"
    if isinstance(precheck_spend, int | float) and precheck_spend > 0:
        sentence += f" ({dash_money(precheck_spend)} of it on the availability check)"
    worst = document.get("worst_case_usd")
    sentence += "."
    if isinstance(worst, int | float):
        sentence += f" The estimate before running, assuming no cache hit, was {dash_money(worst)}."
    return sentence
