"""`endpoints`: list the OpenRouter endpoints serving a model, with prices and health.

httpx and `provibench.bench.openrouter` (pydantic) are imported only inside the callback;
merely importing httpx pulls in rich (it probes for its own bundled CLI extras), so
deferring it keeps building the CLI (schema, --help, completion) cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import click

from provibench.commands.prices import PRICE_COLUMNS
from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    array,
    as_document,
    as_list,
    nullable_boolean,
    nullable_integer,
    nullable_number,
    nullable_string,
    number,
    obj,
    string,
)
from provibench.core.errors import InvalidInput, NotFound, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint

_FETCH_TIMEOUT_S = 30.0
_NOT_FOUND = 404

_PRICES = obj(
    {
        "input": number(),
        "cache_read": number(),
        "cache_write": number(),
        "output": number(),
    },
    required=["input", "cache_read", "cache_write", "output"],
)
_ENDPOINT = obj(
    {
        "provider_name": string(),
        "tag": string(),
        "quantization": nullable_string(),
        "context_length": nullable_integer(),
        "prices": _PRICES,
        "uptime_30m": nullable_number(),
        "uptime_1d": nullable_number(),
        "latency_ms_30m": nullable_number(),
        "throughput_30m": nullable_number(),
        "status": nullable_string(),
        "supports_implicit_caching": nullable_boolean(),
    },
    required=[
        "provider_name",
        "tag",
        "quantization",
        "context_length",
        "prices",
        "uptime_30m",
        "uptime_1d",
        "latency_ms_30m",
        "throughput_30m",
        "status",
        "supports_implicit_caching",
    ],
)
_OUTPUT = obj({"model": string(), "endpoints": array(_ENDPOINT)}, required=["model", "endpoints"])

_SORT_KEYS: dict[str, Callable[[Endpoint], float]] = {
    "price": lambda e: e.prices.input,
    "uptime": lambda e: -(e.uptime_30m if e.uptime_30m is not None else -1.0),
    "latency": lambda e: e.latency_ms_30m if e.latency_ms_30m is not None else float("inf"),
    "throughput": lambda e: -(e.throughput_30m if e.throughput_30m is not None else -1.0),
}

_COLUMNS = (
    "tag",
    "provider",
    "quant",
    "context",
    *PRICE_COLUMNS,
    "uptime 30m",
    "uptime 1d",
    "latency ms",
    "throughput",
    "implicit cache",
)


def render_endpoints(invocation: Invocation, document: Document) -> None:
    """One table: tag, provider, quantization, context, prices, uptime, latency, throughput."""
    from provibench.bench.labels import text_table

    entries = [d for d in map(as_document, as_list(document.get("endpoints")) or []) if d]
    lines = text_table(_COLUMNS, [_endpoint_cells(entry) for entry in entries])
    stdout = invocation.streams.stdout
    stdout.write("\n".join(escape_terminal_text(line) for line in lines) + "\n")
    stdout.flush()


def _cell(value: object) -> str:
    return "-" if value is None else str(value)


def _money(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "-"


def _rate(value: object) -> str:
    return f"{value:.1f}" if isinstance(value, int | float) else "-"


def _endpoint_cells(entry: Document) -> list[str]:
    prices = as_document(entry.get("prices")) or {}
    implicit = entry.get("supports_implicit_caching")
    return [
        _cell(entry.get("tag")),
        _cell(entry.get("provider_name")),
        _cell(entry.get("quantization")),
        _cell(entry.get("context_length")),
        _money(prices.get("input")),
        _money(prices.get("cache_read")),
        _money(prices.get("cache_write")),
        _money(prices.get("output")),
        _rate(entry.get("uptime_30m")),
        _rate(entry.get("uptime_1d")),
        _rate(entry.get("latency_ms_30m")),
        _rate(entry.get("throughput_30m")),
        "yes" if implicit else ("no" if implicit is False else "-"),
    ]


@click.command(
    "endpoints",
    cls=Command,
    spec=CommandSpec(effects=Effects.READ_ONLY, output=_OUTPUT, render=render_endpoints),
    help="List the OpenRouter endpoints serving MODEL, with prices and recent health.\n\n"
    'The listing is fetched with the first kind="openrouter" target\'s API key when its '
    "environment variable is set, because OpenRouter returns the 30-minute latency and "
    "throughput percentiles only for a request that carries a key; without one both are null "
    "(- in the table). This is the same key a sweep's own candidate listing reads, so both "
    "show the same numbers.",
)
@click.argument("model", help="OpenRouter model slug whose endpoints to list")
@click.option(
    "--sort",
    type=click.Choice(sorted(_SORT_KEYS)),
    default="price",
    help="Sort order (price ascending, uptime/throughput descending, latency ascending)",
)
@click.pass_context
def endpoints(ctx: click.Context, model: str, sort: str) -> Document:
    import httpx

    invocation = require_invocation(ctx)
    api_key = _default_api_key(invocation)
    try:
        fetched = asyncio.run(_fetch(model, api_key=api_key))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == _NOT_FOUND:
            raise NotFound(f"OpenRouter lists no model with slug {model!r}") from exc
        raise OperationFailed(f"Fetching endpoints for {model!r} failed: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OperationFailed(f"Fetching endpoints for {model!r} failed: {exc}") from exc
    ordered = sorted(fetched, key=_SORT_KEYS[sort])
    return {"model": model, "endpoints": [_endpoint_document(e) for e in ordered]}


async def _fetch(model: str, *, api_key: str | None) -> list[Endpoint]:
    import httpx

    from provibench.bench.openrouter import fetch_endpoints

    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
        # The unkeyed call stays a two-argument one, the same convention
        # `bench.estimate.fetch_endpoint_index` uses, so a test double with the plain
        # `(client, model)` signature keeps working when no key is configured.
        if api_key is None:
            return await fetch_endpoints(client, model)
        return await fetch_endpoints(client, model, api_key=api_key)


def _default_api_key(invocation: Invocation) -> str | None:
    """The first configured OpenRouter target's key, best-effort, or `None` without one.

    `endpoints` takes no --target, so it reads the key the way sweep's default (--sort
    price) listing would: the first kind="openrouter" target's own environment variable,
    when it is set. OpenRouter returns latency_ms_30m and throughput_30m only for a keyed
    request, so sharing this resolution with sweep is what makes `endpoints` show the same
    numbers the sweep candidate table does, instead of nulls from an unauthenticated call.
    """
    from provibench.commands.run_specs import gateway_target, load_targets

    try:
        targets = load_targets(invocation)
        gateway = gateway_target(None, targets)
    except InvalidInput:
        return None
    return invocation.env.get(gateway.api_key_env) or None


def _endpoint_document(endpoint: Endpoint) -> Document:
    return {
        "provider_name": endpoint.provider_name,
        "tag": endpoint.tag,
        "quantization": endpoint.quantization,
        "context_length": endpoint.context_length,
        "prices": {
            "input": endpoint.prices.input,
            "cache_read": endpoint.prices.cache_read,
            "cache_write": endpoint.prices.cache_write,
            "output": endpoint.prices.output,
        },
        "uptime_30m": endpoint.uptime_30m,
        "uptime_1d": endpoint.uptime_1d,
        "latency_ms_30m": endpoint.latency_ms_30m,
        "throughput_30m": endpoint.throughput_30m,
        "status": None if endpoint.status is None else str(endpoint.status),
        "supports_implicit_caching": endpoint.supports_implicit_caching,
    }
