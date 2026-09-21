"""Tests for promptreplay.bench.probe_pricing: pricing an unpinned spec by what served it.

`tests/fixtures/probe_served_pricing.jsonl` is a trimmed copy of a real run's own jsonl (a
tester's `promptreplay probe` of the packaged sample trace against an unpinned OpenRouter
spec) -- ten requests, every one served by GMICloud and carrying OpenRouter's own billed
`generation.total_cost`. No trace, API key, or private data is in it. That fixture predates
`listing_prices` (`bench.probe_runs`), so it is also the fallback path's own test: every call
below that reads it passes no `listing`, `served_prices`/`choose_prices` default to `None`,
and the fit is exercised exactly as an old, already-persisted run would still exercise it.
"""

from __future__ import annotations

from pathlib import Path

from promptreplay.bench.estimate import SpecPrices
from promptreplay.bench.probe_models import ProbeResult
from promptreplay.bench.probe_pricing import choose_prices, served_prices
from promptreplay.bench.probe_summary import summarize_probe
from promptreplay.bench.targets import Prices

_FIXTURES = Path(__file__).parent / "fixtures"
_WORST_CASE_PRICE = 0.155
"""Venice's listed price at plan time -- the worst case the run's own `run.json` recorded."""


def _fixture(name: str) -> list[ProbeResult]:
    lines = (_FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return [ProbeResult.model_validate_json(line) for line in lines if line.strip()]


def _worst_case() -> SpecPrices:
    return SpecPrices(
        prices=Prices(input=_WORST_CASE_PRICE, cache_read=0.0075, cache_write=0.0, output=1.5),
        source="openrouter-endpoint",
        provider="Venice",
    )


def _served(provider: str, *, prompt: int, cached: int = 0) -> ProbeResult:
    return ProbeResult(
        spec_label="or:model",
        rung=1,
        role="warm",
        attempt=1,
        seq=1,
        status=200,
        latency_ms=10.0,
        provider=provider,
        prompt_total=prompt,
        cached=cached,
    )


def _listed(input_price: float, cache_read: float = 0.0) -> Prices:
    return Prices(input=input_price, cache_read=cache_read, cache_write=0.0, output=0.5)


def test_served_prices_fits_the_one_provider_that_answered() -> None:
    """Item 2 fallback: every request was served by GMICloud, so the fit recovers its rates."""
    records = _fixture("probe_served_pricing.jsonl")

    fit = served_prices(records)

    assert fit is not None
    prices, providers, is_listed = fit
    assert providers == ["GMICloud"]
    assert prices.provider == "GMICloud"
    assert prices.source == "measured"
    assert is_listed is False


def test_choose_prices_reprices_an_unpinned_spec_by_what_served_it() -> None:
    """Item 2 fallback: an unpinned spec's `priced_as` names what served it, not the worst
    case, and the effective price this implies is near the run's own billed cost (real:
    ~0.117 $/M), not Venice's listed 0.155 -- the worst-case endpoint that never answered.
    O1: the fit is noisier than a listing, so `listed` (the third return value, what
    `history`/`compare` read) stays the worst case unchanged, not the fitted number."""
    records = _fixture("probe_served_pricing.jsonl")

    priced, priced_as, listed = choose_prices(_worst_case(), records, unpinned_gateway=True)

    assert priced_as == "served: GMICloud (rates fitted from this run's billed records)"
    assert priced is not None
    assert priced.prices.input != _WORST_CASE_PRICE
    assert listed is not None
    assert listed.prices.input == _WORST_CASE_PRICE


def test_a_pinned_spec_keeps_its_listed_price_unchanged() -> None:
    """Item 2: repricing is only for an unpinned OpenRouter spec; a pinned or native spec's
    price is exactly known in advance and `choose_prices` must not touch it."""
    records = _fixture("probe_served_pricing.jsonl")

    priced, priced_as, listed = choose_prices(_worst_case(), records, unpinned_gateway=False)

    assert priced_as == "pinned"
    assert priced is not None
    assert priced.prices.input == _WORST_CASE_PRICE
    assert listed is priced


def test_summarize_probe_reports_the_served_price_and_priced_as() -> None:
    """The acceptance case end to end: `report`'s own summary carries `priced_as` and an
    effective price near the run's real billed rate, not the worst case; `listed_input` is
    the worst case unchanged, since a fitted reprice is not a listed price."""
    records = _fixture("probe_served_pricing.jsonl")

    summary = summarize_probe(
        "openrouter:deepseek/deepseek-v4.1-flash",
        records,
        prices=_worst_case(),
        unpinned_gateway=True,
    )

    assert summary.priced_as == "served: GMICloud (rates fitted from this run's billed records)"
    assert f"priced as {summary.priced_as}" in summary.notes
    assert summary.eff_per_m_prompt is not None
    assert 0.10 < summary.eff_per_m_prompt < 0.13
    assert summary.listed_input == _WORST_CASE_PRICE
    assert summary.listed_source == "openrouter-endpoint"


def test_served_prices_uses_the_runs_own_listing_over_fitting_it() -> None:
    """O1: a run with a `listing_prices` block (even from a single-request spec that could
    never pin an OLS fit down) reprices from the listing, not from the billed records."""
    records = [_served("GMICloud", prompt=100, cached=0)]
    listing = {"gmicloud": _listed(0.285, 0.0057)}

    fit = served_prices(records, listing)

    assert fit is not None
    prices, providers, is_listed = fit
    assert is_listed is True
    assert providers == ["GMICloud"]
    assert prices.source == "listing"
    assert prices.prices.input == 0.285
    assert prices.prices.cache_read == 0.0057


def test_choose_prices_names_a_single_listed_provider() -> None:
    """O1's exact wording for one listed provider: `served: <name> (listed)`."""
    records = [_served("GMICloud", prompt=100)]
    listing = {"gmicloud": _listed(0.285)}

    priced, priced_as, listed = choose_prices(
        _worst_case(), records, unpinned_gateway=True, listing=listing
    )

    assert priced_as == "served: GMICloud (listed)"
    assert priced is not None
    assert priced.prices.input == 0.285
    assert listed is priced  # a listed reprice is authoritative, not fit noise


def test_served_prices_combines_two_listed_providers_by_weight() -> None:
    """O1's weighted path: two providers' listed rates combine by the prompt tokens each
    one served, heaviest first in the returned provider order."""
    records = [
        _served("GMICloud", prompt=300),
        _served("Venice", prompt=100),
    ]
    listing = {"gmicloud": _listed(0.30), "venice": _listed(0.10)}

    fit = served_prices(records, listing)

    assert fit is not None
    prices, providers, is_listed = fit
    assert is_listed is True
    assert providers == ["GMICloud", "Venice"]
    # weight 300 at 0.30 and 100 at 0.10, over 400 total
    assert prices.prices.input == (300 * 0.30 + 100 * 0.10) / 400


def test_choose_prices_names_weighted_listed_providers() -> None:
    """O1's exact wording for several listed providers: `(listed, weighted)`."""
    records = [_served("GMICloud", prompt=300), _served("Venice", prompt=100)]
    listing = {"gmicloud": _listed(0.30), "venice": _listed(0.10)}

    _priced, priced_as, _listed_prices = choose_prices(
        _worst_case(), records, unpinned_gateway=True, listing=listing
    )

    assert priced_as == "served: GMICloud, Venice (listed, weighted)"


def test_served_prices_with_a_listing_present_never_falls_back_to_fitting() -> None:
    """O1: once a run has its own listing, an unmatched served provider is unpriced rather
    than fitted -- `served_prices` returns `None`, and `choose_prices` reads worst case."""
    records = [_served("Unlisted", prompt=100, cached=0)]
    listing = {"gmicloud": _listed(0.285)}

    assert served_prices(records, listing) is None

    worst_case = _worst_case()
    priced, priced_as, listed = choose_prices(
        worst_case, records, unpinned_gateway=True, listing=listing
    )
    assert priced_as == "worst case: Venice"
    assert priced is worst_case
    assert listed is worst_case


def test_a_fit_far_past_the_worst_case_is_rejected_as_noise() -> None:
    """O1's added guard: a fitted input rate more than 10x the worst-case listed rate is
    noise, not a price -- the caller's worst case stands instead, the way a fit that could
    not be pinned down at all already does.

    Two records with a different, independently varying uncached/output mix pin an exact
    OLS solution of $10,000/M input -- the fit is mathematically exact, not a bad
    convergence, which is exactly why an upper bound (not a residual check) is the guard
    that catches it.
    """
    records = [
        _fit_row(uncached=100, output=200, cost=1.0001),
        _fit_row(uncached=50, output=800, cost=0.5004),
    ]

    worst_case = _worst_case()
    priced, priced_as, listed = choose_prices(worst_case, records, unpinned_gateway=True)

    assert priced_as == "worst case: Venice"
    assert priced is worst_case
    assert listed is worst_case


def _fit_row(*, uncached: int, output: int, cost: float) -> ProbeResult:
    """One served, billed record with an exact token mix an OLS fit needs to solve for."""
    from promptreplay.bench.trace import Usage

    return ProbeResult(
        spec_label="or:model",
        rung=1,
        role="warm",
        attempt=1,
        seq=1,
        status=200,
        latency_ms=10.0,
        provider="Noisy",
        prompt_total=uncached,
        cached=0,
        usage=Usage(output_tokens=output),
        generation={"total_cost": cost},
    )
