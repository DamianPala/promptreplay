"""`sweep`: probe every OpenRouter endpoint of a model, plus the native targets carrying it.

One command answers "which reseller should I use for this model this week": the endpoint
list of the model becomes one pinned spec per tag, each native target that aliases the
model becomes one more spec, and the whole list runs through the probe's own flow — the
same estimate, the same confirmation, the same `runs/<trace>/<timestamp>/` record and the
same tables, ordered by the effective price the run measured.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from provibench.commands.inspect import resolve_trace_path
from provibench.commands.probe import execute_probe, output_fields
from provibench.commands.probe_flags import ProbeRequest, probe_options
from provibench.commands.run_specs import (
    fetch_model_endpoints,
    gateway_target,
    load_targets,
    sweep_specs,
    untagged_count,
)
from provibench.commands.summary_view import render_probe_run
from provibench.core.documents import Document, array, obj, string
from provibench.core.errors import OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint

_SWEEP = obj(
    {
        "model": string(),
        "target": string(),
        "included": array(string()),
        "excluded": array(string()),
    },
    required=["model", "target", "included", "excluded"],
)
_PROPERTIES, _REQUIRED = output_fields()
_OUTPUT = obj({**_PROPERTIES, "sweep": _SWEEP}, required=[*_REQUIRED, "sweep"])


@click.command(
    "sweep",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.NON_IDEMPOTENT, confirm=True, output=_OUTPUT, render=render_probe_run
    ),
    help="Probe every OpenRouter endpoint of MODEL, plus the native targets that carry it.\n\n"
    "MODEL is the OpenRouter slug (deepseek/deepseek-v4.1-flash). The command lists the "
    "model's endpoints, pins one probe spec to each tag, and adds a spec for every native "
    "target whose [targets.<name>.aliases] maps that slug to one of its own model names. "
    "The run is then a probe: the same estimate and --budget, the same confirmation, the "
    "same run directory and tables, ordered by the effective price it measured.",
)
@click.argument("trace")
@click.argument("model")
@click.option(
    "--target",
    default=None,
    help='The kind="openrouter" target to sweep; defaults to the first one in targets.toml',
)
@click.option(
    "--include",
    default=[],
    multiple=True,
    help="Keep only endpoints whose tag starts with this; repeatable, unknown tags error",
)
@click.option(
    "--exclude",
    default=[],
    multiple=True,
    help="Drop endpoints whose tag starts with this; repeatable, unknown tags error",
)
@click.option(
    "--parallel",
    type=click.IntRange(1),
    default=1,
    show_default=True,
    help="Specs to probe at once; one spec is always sequential, and latency numbers get "
    "noisier as this rises",
)
@probe_options
@click.pass_context
def sweep(  # noqa: PLR0913 (click binds one parameter per flag; there is no group to extract)
    ctx: click.Context,
    *,
    trace: str,
    model: str,
    target: str | None,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    parallel: int,
    rungs: str | None,
    repeats: str,
    gap: float,
    warm: bool,
    ttl: str | None,
    no_throughput: bool,
    conversation: str | None,
    strip_thinking: bool,
    timeout: float,
    budget: float | None,
    yes: bool,
) -> Document:
    from provibench.bench.probe_runs import SweepInfo

    invocation = require_invocation(ctx)
    targets = load_targets(invocation)
    gateway = gateway_target(target, targets)
    # A mistyped TRACE is the cheapest mistake to make, so it fails before the one network
    # call this command makes on its own; `execute_probe` resolves the same path again.
    resolve_trace_path(trace, invocation)
    index, endpoints = _endpoint_list(model)
    untagged = untagged_count(endpoints)
    if untagged:
        invocation.message(f"{untagged} endpoint(s) of {model} came back without a tag; skipped")
    sweep_info = SweepInfo(
        model=model,
        target=gateway.name,
        included=list(include),
        excluded=list(exclude),
    )
    return execute_probe(
        invocation,
        ProbeRequest(
            trace=trace,
            specs=sweep_specs(model, gateway, targets, endpoints, include=include, exclude=exclude),
            conversation=conversation,
            rungs=rungs,
            repeats=repeats,
            gap=gap,
            warm=warm,
            ttl=ttl,
            throughput=not no_throughput,
            strip_thinking=strip_thinking,
            timeout=timeout,
            budget=budget,
            yes=yes,
            parallel=parallel,
            by_price=True,
            endpoints=index,
            sweep=sweep_info,
        ),
    )


def _endpoint_list(model: str) -> tuple[dict[str, list[Endpoint]], list[Endpoint]]:
    """The model's OpenRouter endpoints: the run's snapshot and the list specs are built from.

    A sweep without that list is not a sweep, so a failed or empty lookup fails the command
    here — before a spec is built, an estimate printed or an API key asked for.
    """
    index, notes = fetch_model_endpoints([model])
    endpoints = index.get(model, [])
    if not endpoints:
        raise OperationFailed(
            notes[0] if notes else f"No OpenRouter endpoint serves the model {model!r}",
            hint="Check the slug with: provibench endpoints MODEL",
        )
    return index, endpoints
