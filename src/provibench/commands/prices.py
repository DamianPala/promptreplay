"""`prices`: what a native target's models cost, where each price came from, and how fresh.

Every price this tool reports has a source: an OpenRouter endpoint, a `targets.toml` `prices`
entry, or LiteLLM's community table. The first two are stated by the user, the third is
fetched and cached, and this command is where all three can be read before a run is paid for
— the caveat that the community table lists peak rates included.

It also owns the cache for the runs that price a native spec: `native_price_table` is what
`probe` and `replay` call before their estimate, so a run fetches the table at most once and
never while a fresh copy is on disk.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
functions that use them, so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import click

from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    array,
    as_document,
    as_list,
    boolean,
    nullable_number,
    nullable_string,
    obj,
    string,
)
from provibench.core.errors import OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.prices import Cached, PriceTable
    from provibench.bench.targets import RunSpec, Target

_ROW = obj(
    {
        "target": string(),
        "model": string(),
        "source": string(),
        "input": nullable_number(),
        "cache_read": nullable_number(),
        "cache_write": nullable_number(),
        "output": nullable_number(),
    },
    required=["target", "model", "source", "input", "cache_read", "cache_write", "output"],
)
_CACHE_FIELDS = {
    "cache_path": string(),
    "source_url": string(),
    "cache_age_s": nullable_number(),
    "cache_age": nullable_string(),
    "fetched": boolean(),
    "note": nullable_string(),
    "prices": array(_ROW),
    "changed": boolean(),
}
_OUTPUT = obj(_CACHE_FIELDS, required=list(_CACHE_FIELDS))

PRICE_COLUMNS = ("in", "cache\nread", "cache\nwrite", "out")
"""The four price headers shared by the `prices` and `endpoints` tables."""

_COLUMNS = ("target", "model", "source", *PRICE_COLUMNS)

_NOT_PRICED = "n/a"
"""No `targets.toml` entry and no LiteLLM key: the run is priced in tokens only."""


def render_prices(invocation: Invocation, document: Document) -> None:
    """One table: every native target model, its prices, and where they came from."""
    from rich import box
    from rich.markup import escape
    from rich.table import Table

    def cell(value: object) -> str:
        return escape(escape_terminal_text("-" if value is None else str(value)))

    def money(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, int | float) else "-"

    console = invocation.stdout_console()
    caption = (
        f"price table ($/M): {document.get('cache_path')}, fetched {document.get('cache_age')}"
    )
    console.print(escape_terminal_text(caption), highlight=False)

    table = Table(box=box.SIMPLE, header_style="bold")
    for column in _COLUMNS:
        table.add_column(column)
    for entry in _documents(document):
        table.add_row(
            cell(entry.get("target")),
            cell(entry.get("model")),
            cell(entry.get("source")),
            money(entry.get("input")),
            money(entry.get("cache_read")),
            money(entry.get("cache_write")),
            money(entry.get("output")),
        )
    console.print(table)


@click.command(
    "prices",
    cls=Command,
    spec=CommandSpec(effects=Effects.IDEMPOTENT, output=_OUTPUT, render=render_prices),
    help="Show the prices of the native targets' models and where each one comes from.\n\n"
    "A native target is priced from its targets.toml prices entry when it has one (source "
    "table) and from LiteLLM's community table otherwise (source litellm); a model neither "
    "lists is n/a. Without MODEL, every model the native targets name — their prices keys "
    "and the models their aliases map to — is shown. The community table is cached under "
    "$XDG_CACHE_HOME/provibench/litellm-prices.json and refetched at most once a week, or "
    "now with --update; its age is printed beside the path. It lists peak rates, so a "
    "provider that discounts off-peak needs that rate in targets.toml to be priced right.",
)
@click.option("--update", is_flag=True, help="Refetch the LiteLLM price table before reading it")
@click.argument("models", nargs=-1, help="Model names to price for every native target")
@click.pass_context
def prices(ctx: click.Context, *, update: bool, models: tuple[str, ...]) -> Document:
    invocation = require_invocation(ctx)
    from provibench.bench.prices import URL, TableUnavailable, format_age
    from provibench.commands.run_specs import load_targets

    targets = load_targets(invocation)
    try:
        state = cached_prices(invocation, update=update)
    except TableUnavailable as exc:
        raise OperationFailed(str(exc), hint=exc.hint) from exc
    rows = [
        _row(target, model, state.table)
        for target in _native(targets)
        for model in _models_of(target, models)
    ]
    return {
        "cache_path": str(state.path),
        "source_url": URL,
        "cache_age_s": state.age_s,
        "cache_age": format_age(state.age_s),
        "fetched": state.fetched,
        "note": state.warning,
        "prices": rows,
        "changed": state.fetched,
    }


def cached_prices(invocation: Invocation, *, update: bool = False) -> Cached:
    """The LiteLLM table for this invocation, refetched when the cache is stale or `update`.

    A failed refresh while a copy is on disk is a warning and that copy; it is what keeps a
    run working offline for a week. With no copy there is nothing to price with, which
    `TableUnavailable` says with the URL and the override that prices a model without the
    table at all.
    """
    from provibench.bench.prices import cache_path, cached_table

    path = cache_path(invocation.env, invocation.home)
    state = cached_table(path, update=update)
    if state.warning is not None:
        invocation.message(state.warning)
    return state


def native_price_table(invocation: Invocation, specs: Sequence[RunSpec]) -> PriceTable | None:
    """The community table when a native spec has no `targets.toml` price, else `None`.

    A run that prices every native spec from `targets.toml` never reads the cache or the
    network. One that needs the community table gets it here, once, before the estimate:
    the estimate is the number the caller is asked to confirm, so it is priced before it is
    rendered and the line saying the table was fetched lands above it.

    A table that cannot be had is not fatal to a run. `prices` is the command whose whole
    job is the table, and it refuses; a probe or a replay priced as `n/a` is what it did
    before this table existed, and `--budget` is what refuses to spend on an unpriced spec.
    So the run says in one line why the prices are missing and carries on, and a fresh copy
    on disk never reaches this branch at all.
    """
    if not any(needs_litellm(spec) for spec in specs):
        return None
    from provibench.bench.prices import TableUnavailable

    try:
        state = cached_prices(invocation)
    except TableUnavailable as exc:
        invocation.message(exc.line)
        return None
    if state.fetched:
        invocation.message(f"fetched the LiteLLM price table into {state.path}")
    return state.table


def needs_litellm(spec: RunSpec) -> bool:
    """Whether pricing this spec needs the community table: native, and not already stated."""
    # the early-out is only right because `resolve_target` prefers the targets.toml entry;
    # change the precedence there and this test has to follow
    return spec.target.kind != "openrouter" and spec.model not in spec.target.prices


def _native(targets: Mapping[str, Target]) -> list[Target]:
    """The targets priced from a table rather than from an endpoint list, in file order."""
    return [target for target in targets.values() if target.kind != "openrouter"]


def _models_of(target: Target, models: Sequence[str]) -> list[str]:
    """The models to price for one target: the ones asked for, else the target's own.

    A target's own are the models its `prices` table names and the ones its `aliases` map to
    a slug; together they are every model a run spec can hand this target without inventing
    a name, which is what makes this the list worth checking after editing targets.toml.
    """
    if models:
        return list(dict.fromkeys(models))
    return sorted({*target.prices, *target.aliases.values()})


def _row(target: Target, model: str, table: PriceTable) -> Document:
    """One model's prices and source, with every price `null` when nothing prices it."""
    from provibench.bench.prices import resolve_target

    found = resolve_target(target, model, table)
    prices = None if found is None else found[0]
    return {
        "target": target.name,
        "model": model,
        "source": _NOT_PRICED if found is None else found[1],
        "input": None if prices is None else prices.input,
        "cache_read": None if prices is None else prices.cache_read,
        "cache_write": None if prices is None else prices.cache_write,
        "output": None if prices is None else prices.output,
    }


def _documents(document: Document) -> list[Document]:
    """The `prices` array of a document, as documents."""
    return [entry for entry in map(as_document, as_list(document.get("prices")) or []) if entry]
