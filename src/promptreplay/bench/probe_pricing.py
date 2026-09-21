"""Pricing an unpinned spec by what actually served it, once the run has happened.

An unpinned OpenRouter spec can be served by any endpoint behind the model, so the estimate
prices it at the most expensive one as a worst case. Once the run has happened,
`providers_seen` says who actually answered, and the honest reprice is that provider's own
listed rate. A run persists its own endpoint listing as `listing_prices` (see
`bench.probe_runs.listing_prices`), so `served_prices` looks the answering provider up there
first; several providers combine by a prompt-token-weighted mean, the same way the estimate
weighs a mixed run.

A run written before `listing_prices` existed carries none (`None`, not an empty mapping),
and only then does `served_prices` fall back to fitting the provider's rates from its own
billed records (`ProbeResult.generation`) -- ordinary least squares through the origin,
input and output first, then cache-read from what those two do not explain. The fit is
noisier than a listing (see `_fit`'s docstring), so a fitted rate more than
`_FIT_UPPER_BOUND_MULTIPLE` times the run's own worst-case listed rate is rejected as noise,
not a price, and the caller's worst case stands instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from promptreplay.bench.estimate import SpecPrices
from promptreplay.bench.openrouter import normalize_provider
from promptreplay.bench.probe_models import ProbeResult, is_served
from promptreplay.bench.targets import Prices

__all__ = ["choose_prices", "served_prices"]

_MEASURED_SOURCE = "measured"
_LISTING_SOURCE = "listing"
_FITTED = "rates fitted from this run's billed records"
"""Said in `priced_as` itself: these rates were reconstructed, not read from a listing, and
a reader comparing them against `prices` or `endpoints` is owed that before they do."""
_FIT_UPPER_BOUND_MULTIPLE = 10
"""A fitted input rate past this multiple of the worst-case listed rate is noise, not a
price: with as few as two billed records the fit is exact and unbounded (see `_fit`)."""


@dataclass(frozen=True, slots=True)
class _Row:
    """One billed record, reduced to what the fit needs: its token mix and its cost."""

    uncached: int
    output: int
    cached: int
    cost: float


def choose_prices(
    prices: SpecPrices | None,
    records: Sequence[ProbeResult],
    *,
    unpinned_gateway: bool,
    listing: Mapping[str, Prices] | None = None,
) -> tuple[SpecPrices | None, str, SpecPrices | None]:
    """The prices `eff_per_m_prompt` is computed from, how they were chosen, and the prices
    `history`/`compare` should treat as this run's listed price.

    Returns `(eff_prices, priced_as, listed_prices)`. A pinned or native spec's price is
    exactly known in advance and both tuple entries are `prices`, unchanged. An unpinned
    OpenRouter spec's estimate priced the most expensive endpoint as a worst case; once
    served, `served_prices` reprices it from what actually answered, and `eff_prices` is
    that -- not the worst case -- so `eff_per_m_prompt` reflects it. `listed_prices` is the
    same only when the reprice came from the run's own listing (an authoritative number, not
    noise); a fitted reprice, or none at all, leaves `listed_prices` at the original worst
    case, so two runs of the same spec never show `history`/`compare` a price change that is
    fit noise (see the module docstring).
    """
    if not unpinned_gateway:
        return prices, "pinned", prices
    found = served_prices(records, listing)
    if found is not None:
        served, providers, is_listed = found
        weighted = len(providers) > 1
        if is_listed:
            return served, _listed_note(served.provider, weighted=weighted), served
        if _within_bound(served, prices):
            return served, _fitted_note(served.provider, weighted=weighted), prices
    provider = prices.provider if prices is not None else None
    label = f"worst case: {provider}" if provider else "worst case: n/a"
    return prices, label, prices


def served_prices(
    records: Sequence[ProbeResult], listing: Mapping[str, Prices] | None = None
) -> tuple[SpecPrices, list[str], bool] | None:
    """The prices implied by what each served provider actually billed, or `None`.

    `listing` present (even empty) means the run recorded its own endpoint listing: the
    served providers are looked up there and nothing is fitted, even when none of them
    match. `listing` `None` means an older run, and the served providers' rates are fit from
    their own billed records instead. One provider's rates are used as they are; several
    combine by a prompt-token-weighted mean of what each one served. Returns the combined
    `SpecPrices` (`provider` the joined provider names), the providers it drew from, heaviest
    first, and whether they came from the listing (`True`) or a fit (`False`).
    """
    if listing is not None:
        return _listed_prices(records, listing)
    return _fitted_prices(records)


def _listed_prices(
    records: Sequence[ProbeResult], listing: Mapping[str, Prices]
) -> tuple[SpecPrices, list[str], bool] | None:
    """Each served provider's own listed rates, weighted by the prompt tokens it served."""
    weights: dict[str, int] = defaultdict(int)
    for record in records:
        if is_served(record) and record.provider:
            weights[record.provider] += record.prompt_total
    matched = {
        provider: (listing[key], weight)
        for provider, weight in weights.items()
        if (key := normalize_provider(provider)) in listing
    }
    if not matched:
        return None
    ordered = sorted(matched, key=lambda name: matched[name][1], reverse=True)
    if len(ordered) == 1:
        [name] = ordered
        listed, _weight = matched[name]
        return SpecPrices(prices=listed, source=_LISTING_SOURCE, provider=name), ordered, True
    total_weight = sum(weight for _, weight in matched.values())
    combined = Prices(
        input=_weighted(matched, total_weight, lambda p: p.input),
        cache_read=_weighted(matched, total_weight, lambda p: p.cache_read),
        cache_write=_weighted(matched, total_weight, lambda p: p.cache_write),
        output=_weighted(matched, total_weight, lambda p: p.output),
    )
    names = ", ".join(sorted(ordered))
    return SpecPrices(prices=combined, source=_LISTING_SOURCE, provider=names), ordered, True


def _weighted(
    matched: Mapping[str, tuple[Prices, int]], total_weight: int, field: Callable[[Prices], float]
) -> float:
    return sum(field(prices) * weight for prices, weight in matched.values()) / total_weight


def _fitted_prices(records: Sequence[ProbeResult]) -> tuple[SpecPrices, list[str], bool] | None:
    """Each served provider's rates reconstructed by fitting its own billed records."""
    by_provider: dict[str, list[_Row]] = defaultdict(list)
    for record in records:
        cost = _billed(record)
        if is_served(record) and record.provider and cost is not None:
            uncached = max(record.prompt_total - record.cached, 0)
            row = _Row(uncached, record.usage.output_tokens, record.cached, cost)
            by_provider[record.provider].append(row)

    fitted: dict[str, tuple[Prices, int]] = {}
    for provider, rows in by_provider.items():
        prices = _fit(rows)
        if prices is not None:
            fitted[provider] = (prices, sum(row.uncached + row.cached for row in rows))

    if not fitted:
        return None
    ordered = sorted(fitted, key=lambda name: fitted[name][1], reverse=True)
    if len(ordered) == 1:
        [name] = ordered
        prices, _weight = fitted[name]
        return SpecPrices(prices=prices, source=_MEASURED_SOURCE, provider=name), ordered, False
    total_weight = sum(weight for _, weight in fitted.values())
    input_price = sum(prices.input * weight for prices, weight in fitted.values()) / total_weight
    cache_read = (
        sum(prices.cache_read * weight for prices, weight in fitted.values()) / total_weight
    )
    combined = Prices(input=input_price, cache_read=cache_read, cache_write=0.0, output=0.0)
    names = ", ".join(sorted(ordered))
    return SpecPrices(prices=combined, source=_MEASURED_SOURCE, provider=names), ordered, False


def _within_bound(fitted: SpecPrices, worst_case: SpecPrices | None) -> bool:
    """A fitted input rate is accepted unless it is far past the run's own worst case.

    With nothing to compare against (no worst-case price on hand) the fit is accepted as is;
    that only happens when the estimate itself could not price the spec, which is rarer than
    a noisy fit and not this guard's job to second-guess.
    """
    if worst_case is None or worst_case.prices.input <= 0:
        return True
    return fitted.prices.input <= worst_case.prices.input * _FIT_UPPER_BOUND_MULTIPLE


def _fitted_note(provider: str | None, *, weighted: bool) -> str:
    prefix = "weighted, " if weighted else ""
    return f"served: {provider or 'n/a'} ({prefix}{_FITTED})"


def _listed_note(provider: str | None, *, weighted: bool) -> str:
    suffix = ", weighted" if weighted else ""
    return f"served: {provider or 'n/a'} (listed{suffix})"


def _fit(rows: Sequence[_Row]) -> Prices | None:
    """One provider's implied `Prices`, or `None` when its own records cannot pin them down."""
    solved = _solve_input_output(rows)
    if solved is None:
        return None
    input_rate, output_rate = solved
    cache_read_rate = _solve_cache_read(rows, input_rate, output_rate)
    return Prices(
        input=input_rate * 1e6,
        cache_read=cache_read_rate * 1e6,
        cache_write=0.0,
        output=output_rate * 1e6,
    )


def _solve_input_output(rows: Sequence[_Row]) -> tuple[float, float] | None:
    """USD-per-token input and output rates from `cost = input*uncached + output*out`.

    Ordinary least squares through the origin over every billed record of the provider;
    `None` when the records do not vary enough to separate the two rates (a zero
    determinant), or when the fit implies an input rate of zero or less -- a provider that
    billed nothing, or noise the data is too thin to tell from a real rate. Either way the
    caller's worst case stands, which is wrong by a knowable margin rather than free.
    """
    sxx = sum(row.uncached * row.uncached for row in rows)
    sxy = sum(row.uncached * row.output for row in rows)
    syy = sum(row.output * row.output for row in rows)
    sxz = sum(row.uncached * row.cost for row in rows)
    syz = sum(row.output * row.cost for row in rows)
    determinant = sxx * syy - sxy * sxy
    if determinant == 0:
        return None
    input_rate = (sxz * syy - syz * sxy) / determinant
    output_rate = (sxx * syz - sxy * sxz) / determinant
    if input_rate <= 0 or output_rate < 0:
        return None
    return input_rate, output_rate


def _solve_cache_read(rows: Sequence[_Row], input_rate: float, output_rate: float) -> float:
    """The cache-read rate left once the input and output rates are paid for.

    A single-unknown least-squares fit through the origin: `residual = cache_read * cached`,
    where `residual` is what the input and output rates do not explain. Zero when no record
    of this provider ever read from the cache -- there is nothing to fit a rate to.
    """
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        if row.cached <= 0:
            continue
        residual = row.cost - input_rate * row.uncached - output_rate * row.output
        numerator += residual * row.cached
        denominator += row.cached * row.cached
    if denominator <= 0:
        return 0.0
    return max(numerator / denominator, 0.0)


def _billed(record: ProbeResult) -> float | None:
    value = (record.generation or {}).get("total_cost")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
