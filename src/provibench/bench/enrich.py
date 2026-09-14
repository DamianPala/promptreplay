"""Post-run costing and the OpenRouter `/generation` lookup both protocols share.

Replay and probe send the same bytes and want the same answer back: which provider served,
what the gateway billed, and how many tokens the native endpoint reported. Keeping the
lookup here means a probe record and a replay result are enriched by identical code.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

from provibench.bench.openrouter import (
    Generation,
    fetch_endpoints,
    fetch_generation,
    normalize_provider,
)
from provibench.bench.prices import PriceTable, resolve_target
from provibench.bench.pricing import compute_cost
from provibench.bench.targets import Prices, RunSpec

if TYPE_CHECKING:
    from provibench.bench.replay import ReplayResult


def append_note(existing: str | None, addition: str) -> str:
    """`addition`, appended to whatever note is already on a result."""
    return f"{existing}; {addition}" if existing else addition


def apply_generation(
    result: ReplayResult, gen: Generation | None, price_by_key: dict[str, Prices]
) -> None:
    """Copy one generation lookup onto a result: provider, native counts, billed cost."""
    if gen is None:
        result.note = append_note(result.note, "generation lookup failed")
        return

    result.generation = gen.raw
    result.billed_total = gen.total_cost
    result.cache_discount = gen.cache_discount
    if gen.provider_name:
        result.provider = gen.provider_name
    if gen.native_tokens_prompt is not None:
        result.prompt_total = gen.native_tokens_prompt
    if gen.native_tokens_cached is not None:
        result.cached = gen.native_tokens_cached
    if gen.native_tokens_completion is not None:
        result.output_tokens = gen.native_tokens_completion

    if result.cached > result.prompt_total:
        result.note = append_note(result.note, "cached exceeds prompt_total; input cost skipped")
        return

    prices = price_by_key.get(normalize_provider(gen.provider_name or ""))
    if prices is None:
        note = f"no endpoint price match for provider {gen.provider_name!r}"
        result.note = append_note(result.note, note)
        return

    result.cost = compute_cost(
        input_tokens=result.prompt_total - result.cached,
        cache_read=result.cached,
        cache_write=result.cache_write,
        output_tokens=result.output_tokens,
        prices=prices,
        source="openrouter-endpoint",
    )


async def enrich_openrouter(
    spec: RunSpec, results: list[ReplayResult], client: httpx.AsyncClient, api_key: str
) -> None:
    """Fill each result's provider, native counts, billed cost and raw generation.

    Failures are left to the caller, which decides whether to note them or ignore them.
    """
    endpoints = await fetch_endpoints(client, spec.model)
    price_by_key: dict[str, Prices] = {}
    for ep in endpoints:
        price_by_key[normalize_provider(ep.tag)] = ep.prices
        price_by_key[normalize_provider(ep.provider_name)] = ep.prices

    sem = asyncio.Semaphore(4)
    successful = [r for r in results if r.status == 200 and r.message_id]

    async def _fetch(result: ReplayResult) -> None:
        message_id = result.message_id
        if message_id is None:
            return  # unreachable: `successful` already filtered on message_id
        async with sem:
            gen = await fetch_generation(client, api_key, message_id)
        apply_generation(result, gen, price_by_key)

    await asyncio.gather(*(_fetch(r) for r in successful))


def enrich_anthropic(
    spec: RunSpec, results: list[ReplayResult], table: PriceTable | None = None
) -> None:
    """Cost the results against the target's own price table or the LiteLLM one; no lookup.

    `table` is the resolved community table the run was priced with, so the breakdown the
    run records names the same source the estimate showed before it was paid for. A spec
    with neither price leaves the cost unknown and says so in its note, as it always did.
    """
    found = resolve_target(spec.target, spec.model, table)
    for result in results:
        if result.status != 200:
            continue
        if found is None:
            result.note = append_note(result.note, f"no price table for {spec.model}")
            continue
        result.cost = compute_cost(
            input_tokens=result.usage.input_tokens,
            cache_read=result.cached,
            cache_write=result.cache_write,
            output_tokens=result.output_tokens,
            prices=found[0],
            source=found[1],
        )
