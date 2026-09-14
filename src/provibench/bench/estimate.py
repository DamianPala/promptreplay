"""Worst-case spend estimates: every token a run will send, times the listed input price.

Weak resellers bill close to the no-cache price, so this is the number to show before a run
and the number `--budget` compares against. Prices come from the OpenRouter endpoint list
for gateway specs and from the `targets.toml` price table for native ones; a model with
neither yields a token-only estimate with the price recorded as `n/a`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from provibench.bench.openrouter import normalize_provider
from provibench.bench.probe_models import ProbeOptions
from provibench.bench.probe_stream import STREAM_MAX_TOKENS
from provibench.bench.targets import Prices, RunSpec
from provibench.bench.trace import TraceEntry

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint

type EndpointIndex = dict[str, list[Endpoint]]
"""The OpenRouter endpoint lists of one run, keyed by model id."""

_ENDPOINTS_TIMEOUT_S = 30.0
_LABEL_WIDTH = 48
_HEADERS = ("spec", "tokens", "$/M in", "source", "worst case $")
_NOTE = (
    "worst case assumes no cache hit: every prompt token billed at the listed input price, "
    "excluding retries"
)


class SpecPrices(BaseModel):
    """The listed prices of the endpoint a spec would use, and where they came from."""

    prices: Prices
    source: str
    provider: str | None = None
    """The endpoint's tag or provider name, when the price came from the endpoint list."""


class SpecEstimate(BaseModel):
    """One spec's worst-case spend, before any request is sent."""

    label: str
    tokens: int
    tokens_known: bool = True
    prices: SpecPrices | None = None
    usd: float | None = None
    notes: list[str] = Field(default_factory=list)


async def fetch_endpoint_index(models: Sequence[str]) -> tuple[EndpointIndex, list[str]]:
    """The OpenRouter endpoint list per distinct model, plus one note per failed lookup.

    The endpoint list is both the price source for the estimate and the drift snapshot a
    probe run stores, so a failure is reported and carried on rather than fatal.
    """
    index: EndpointIndex = {}
    notes: list[str] = []
    # Imported here so a test can replace `fetch_endpoints` at its own module.
    from provibench.bench.openrouter import fetch_endpoints

    async with httpx.AsyncClient(timeout=_ENDPOINTS_TIMEOUT_S) as client:
        for model in dict.fromkeys(models):
            try:
                index[model] = await fetch_endpoints(client, model)
            except (httpx.HTTPError, ValueError) as exc:
                notes.append(f"endpoint lookup for {model} failed: {exc}")
    return index, notes


def spec_prices(spec: RunSpec, index: Mapping[str, list[Endpoint]]) -> SpecPrices | None:
    """The listed price a spec would be billed at, or `None` when there is no match.

    A pinned gateway spec takes the price of the endpoint it named. An unpinned one takes
    the most expensive endpoint serving the model: this is the worst case, so the top of the
    ladder is the honest end of it, and the estimate says which endpoint it used.
    """
    if spec.target.kind != "openrouter":
        table = spec.target.prices.get(spec.model)
        return None if table is None else SpecPrices(prices=table, source="table")
    endpoints = index.get(spec.model, [])
    if spec.providers:
        wanted = {normalize_provider(name) for name in spec.providers}
        match = next(
            (
                endpoint
                for endpoint in endpoints
                if normalize_provider(endpoint.tag) in wanted
                or normalize_provider(endpoint.provider_name) in wanted
            ),
            None,
        )
    else:
        match = max(endpoints, key=lambda endpoint: endpoint.prices.input, default=None)
    if match is None:
        return None
    return SpecPrices(
        prices=match.prices,
        source="openrouter-endpoint",
        provider=match.provider_name or match.tag,
    )


def probe_estimate(
    spec: RunSpec,
    entries: Sequence[TraceEntry],
    options: ProbeOptions,
    prices: SpecPrices | None,
) -> SpecEstimate:
    """The worst case for one probe spec: every request it is about to send.

    That is the cold write and the warm reads, the throughput request's prompt plus its
    output budget when it is on, and the TTL re-reads of the first rung. Output tokens are
    priced at the model's output rate, because generation is not prompt tokens and does not
    share their price.
    """
    rungs = options.rungs or []
    requests: list[tuple[int, int]] = []
    output_tokens = 0
    for rung, count in zip(rungs, options.repeats, strict=True):
        requests.append((rung, 1))
        requests.append((rung + 1, count))
        if options.throughput:
            # The streamed generation's prompt is a turn-`k+1` request too; its own 256
            # output tokens are a second cost, billed at the output price.
            requests.append((rung + 1, 1))
            output_tokens += STREAM_MAX_TOKENS
        if options.ttl_s and rung == rungs[0]:
            requests.extend((rung + 1, 1) for _ in options.ttl_s)
    return _estimate(spec, entries, requests, prices, output_tokens=output_tokens)


def replay_estimate(
    spec: RunSpec, entries: Sequence[TraceEntry], prices: SpecPrices | None
) -> SpecEstimate:
    """The worst case for one full replay: every selected turn sent once."""
    requests = [(turn, 1) for turn in range(1, len(entries) + 1)]
    return _estimate(spec, entries, requests, prices)


def _estimate(
    spec: RunSpec,
    entries: Sequence[TraceEntry],
    requests: Sequence[tuple[int, int]],
    prices: SpecPrices | None,
    *,
    output_tokens: int = 0,
) -> SpecEstimate:
    """`requests` are `(turn, times)`; `output_tokens` are the ones the run will generate."""
    tokens = 0
    known = True
    notes: list[str] = []
    for turn, factor in requests:
        recorded = _recorded_tokens(entries[turn - 1])
        if recorded is None:
            known = False
            notes.append(f"turn {turn} has no recorded prompt usage; its tokens are unknown")
            continue
        tokens += recorded * factor
    if prices is None:
        notes.append("no listed input price for this model; the estimate is tokens only")
    elif spec.target.kind == "openrouter" and not spec.providers:
        endpoint = prices.provider or "an unnamed endpoint"
        notes.append(
            f"unpinned: priced from the most expensive endpoint ({endpoint}); "
            "pin @provider to price one endpoint"
        )
    # `tokens` is what the run will send as prompt; the generated tokens are a second cost
    # and are reported as such, not folded into a count that claims to be a prompt size.
    usd = _usd(tokens, output_tokens, prices) if known else None
    return SpecEstimate(
        label=spec.label, tokens=tokens, tokens_known=known, prices=prices, usd=usd, notes=notes
    )


def _usd(prompt_tokens: int, output_tokens: int, prices: SpecPrices | None) -> float | None:
    """The prompt side at the input price, the generated side at the output price.

    A spec with no listed output price is priced at its input price: the throughput request
    is a small part of the estimate, and leaving it out would understate the run.
    """
    if prices is None:
        return None
    output_price = prices.prices.output or prices.prices.input
    return (prompt_tokens * prices.prices.input + output_tokens * output_price) * 1e-6


def _recorded_tokens(entry: TraceEntry) -> int | None:
    """The prompt tokens the recording reports for this turn, or `None` when unrecorded."""
    if entry.response is None:
        return None
    return entry.response.usage.prompt_total


def estimate_total(estimates: Sequence[SpecEstimate]) -> float | None:
    """The sum of every known worst case, or `None` when no spec could be priced."""
    amounts = [estimate.usd for estimate in estimates if estimate.usd is not None]
    return sum(amounts) if amounts else None


def render_estimate(estimates: Sequence[SpecEstimate]) -> str:
    """A fixed-width table of the estimates and their total; `n/a` where unknown."""
    rows = [_estimate_row(estimate) for estimate in estimates]
    total = estimate_total(estimates)
    rows.append(
        [
            "total",
            f"{sum(e.tokens for e in estimates):,}"
            if all(e.tokens_known for e in estimates)
            else "n/a",
            "",
            "",
            f"{total:.4f}" if total is not None else "n/a",
        ]
    )
    widths = [
        max([len(_HEADERS[index]), *(len(row[index]) for row in rows)])
        for index in range(len(_HEADERS))
    ]
    lines = [_join(_HEADERS, widths), _join(["-" * width for width in widths], widths)]
    lines.extend(_join(row, widths) for row in rows)
    lines.append(_NOTE)
    lines.extend(f"note: {note}" for estimate in estimates for note in estimate.notes)
    return "\n".join(lines)


def _estimate_row(estimate: SpecEstimate) -> list[str]:
    price = estimate.prices.prices.input if estimate.prices is not None else None
    return [
        _clip(estimate.label, _LABEL_WIDTH),
        f"{estimate.tokens:,}" if estimate.tokens_known else "n/a",
        f"{price:.3f}" if price is not None else "n/a",
        estimate.prices.source if estimate.prices is not None else "n/a",
        f"{estimate.usd:.4f}" if estimate.usd is not None else "n/a",
    ]


def _clip(cell: str, width: int) -> str:
    return cell if len(cell) <= width else f"{cell[: width - 1]}…"


def _join(cells: Sequence[str], widths: Sequence[int]) -> str:
    return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
