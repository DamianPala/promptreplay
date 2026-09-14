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
from provibench.core.errors import OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint

_FETCH_TIMEOUT_S = 30.0

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
    "$/M in",
    "$/M out",
    "$/M cache read",
    "$/M cache write",
    "uptime 30m",
    "uptime 1d",
    "latency ms",
    "throughput",
    "implicit cache",
)


def render_endpoints(invocation: Invocation, document: Document) -> None:
    """One table: tag, provider, quantization, context, prices, uptime, latency, throughput."""
    from rich import box
    from rich.markup import escape
    from rich.table import Table

    def cell(value: object) -> str:
        return escape(escape_terminal_text("-" if value is None else str(value)))

    def money(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, int | float) else "-"

    def rate(value: object) -> str:
        return f"{value:.1f}" if isinstance(value, int | float) else "-"

    table = Table(box=box.SIMPLE, header_style="bold")
    for column in _COLUMNS:
        table.add_column(column)
    entries = [d for d in map(as_document, as_list(document.get("endpoints")) or []) if d]
    for entry in entries:
        prices = as_document(entry.get("prices")) or {}
        implicit = entry.get("supports_implicit_caching")
        table.add_row(
            cell(entry.get("tag")),
            cell(entry.get("provider_name")),
            cell(entry.get("quantization")),
            cell(entry.get("context_length")),
            money(prices.get("input")),
            money(prices.get("output")),
            money(prices.get("cache_read")),
            money(prices.get("cache_write")),
            rate(entry.get("uptime_30m")),
            rate(entry.get("uptime_1d")),
            rate(entry.get("latency_ms_30m")),
            rate(entry.get("throughput_30m")),
            "yes" if implicit else ("no" if implicit is False else "-"),
        )
    invocation.stdout_console().print(table)


@click.command(
    "endpoints",
    cls=Command,
    spec=CommandSpec(effects=Effects.READ_ONLY, output=_OUTPUT, render=render_endpoints),
    help="List the OpenRouter endpoints serving MODEL, with prices and recent health.",
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

    require_invocation(ctx)
    try:
        fetched = asyncio.run(_fetch(model))
    except httpx.HTTPError as exc:
        raise OperationFailed(f"Fetching endpoints for {model!r} failed: {exc}") from exc
    ordered = sorted(fetched, key=_SORT_KEYS[sort])
    return {"model": model, "endpoints": [_endpoint_document(e) for e in ordered]}


async def _fetch(model: str) -> list[Endpoint]:
    import httpx

    from provibench.bench.openrouter import fetch_endpoints

    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
        return await fetch_endpoints(client, model)


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
