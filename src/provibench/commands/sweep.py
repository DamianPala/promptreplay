"""`sweep`: probe the OpenRouter endpoints a model should be measured on, plus its natives.

One command answers "which reseller should I use for this model this week": the endpoint
list of the model is cut down by the criteria the caller states (`--sort`, `--top`, `--zdr`
over the always-on stability floor), the survivors become the run's candidates, and the run
goes through the probe's own flow — the same estimate, the same confirmation, the same
availability check once it is agreed to, the same `runs/<trace>/<timestamp>/` record and the
same tables, ordered by the effective price the run measured.

The listing is a pre-filter, never a verdict: it says which endpoints were worth paying to
measure and why the others were not, and the report's effective $/M decides between them.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

import click

from provibench.commands.inspect import resolve_trace_path
from provibench.commands.probe import execute_probe, output_fields
from provibench.commands.probe_flags import ProbeRequest, probe_options
from provibench.commands.run_specs import gateway_target, load_targets, native_specs, untagged_count
from provibench.commands.summary_view import SWEEP_BLOCK, render_probe_run
from provibench.commands.sweep_plan import (
    Criteria,
    endpoint_list,
    gateway_key,
    select,
    sweep_info,
)
from provibench.core.documents import Document, obj
from provibench.core.errors import InvalidInput
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

_SORT_KEYS = ("price", "uptime", "throughput", "latency")
"""Every `--sort` key; `bench.selection.SORT_KEYS` owns what each of them ranks by."""

_PROPERTIES, _REQUIRED = output_fields()
_OUTPUT = obj({**_PROPERTIES, "sweep": SWEEP_BLOCK}, required=[*_REQUIRED, "sweep"])


@click.command(
    "sweep",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.NON_IDEMPOTENT,
        confirm=True,
        output=_OUTPUT,
        output_description=(
            "The successful result contains the persisted run directory, the measured summary "
            "for each surviving spec, and sweep selection details. If any request or rung "
            "fails, the same result is available under error.context and the command exits "
            "non-zero."
        ),
        render=render_probe_run,
    ),
    help="Probe the OpenRouter endpoints of MODEL that pass the selection criteria, plus the "
    "native targets that carry it.\n\n"
    "MODEL is the OpenRouter slug (deepseek/deepseek-v4.1-flash). The command lists the "
    "model's endpoints, keeps only the Zero Data Retention list when --zdr, drops the ones "
    "the stability floor excludes (a status below zero, or less than 97 % uptime over the "
    "last day; --include TAG keeps only matching tags and keeps them past the floor, "
    "--exclude TAG drops them), ranks what is left by "
    "--sort (price ascending by default) and keeps the --top N best. Availability is "
    "checked once the run is confirmed: every candidate gets one smallest-rung request, "
    "priced in the estimate you agreed to, so an endpoint the account's settings exclude is "
    "reported as not probed instead of eating a slot. What survives is probed like any "
    "other run — the same estimate and --budget, the same confirmation, the same run "
    "directory and tables. The listing only picks candidates; the effective $/M the report "
    "measures is the verdict.",
)
@click.argument("trace", help="Trace path, name under traces_dir, or 'sample'")
@click.argument("model", help="OpenRouter model slug to sweep")
@click.option(
    "--target",
    default=None,
    help='The kind="openrouter" target to sweep; defaults to the first one in targets.toml',
)
@click.option(
    "--include",
    default=[],
    multiple=True,
    help="Keep only endpoints whose tag starts with this, and keep them past the stability "
    "floor; repeatable, unknown tags error",
)
@click.option(
    "--exclude",
    default=[],
    multiple=True,
    help="Drop endpoints whose tag starts with this; repeatable, unknown tags error",
)
@click.option(
    "--sort",
    "sort_key",
    type=click.Choice(_SORT_KEYS),
    default="price",
    show_default=True,
    help="Ranking key: price ascending (ties to the faster endpoint, then to the one with "
    "the better uptime 1d), uptime 1d descending, throughput p50 descending, or latency "
    "p50 ascending; the last two need the target's API key",
)
@click.option(
    "--top",
    type=click.IntRange(1),
    default=None,
    help="Probe only the N best candidates, after the availability check; native targets "
    "are never cut and count outside N. Also turns the check on",
)
@click.option(
    "--zdr",
    is_flag=True,
    help="Keep only the endpoints on OpenRouter's Zero Data Retention list",
)
@click.option(
    "--check",
    "check",
    is_flag=True,
    help="Send one smallest-rung request per candidate after you confirm the estimate, so a "
    "candidate this key cannot reach is reported instead of probed; priced in that "
    "estimate, and implied by --top",
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
    sort_key: str,
    top: int | None,
    zdr: bool,
    check: bool,
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
    invocation = require_invocation(ctx)
    targets = load_targets(invocation)
    gateway = gateway_target(target, targets)
    # A mistyped TRACE is the cheapest mistake to make, so it fails before the first round
    # trip; `execute_probe` resolves the same path again for the run itself.
    resolve_trace_path(trace, invocation)
    index, endpoints = endpoint_list(model, api_key=gateway_key(sort_key, gateway, invocation))
    untagged = untagged_count(endpoints)
    if untagged:
        invocation.message(f"{untagged} endpoint(s) of {model} came back without a tag; skipped")
    selection = select(
        model,
        gateway,
        endpoints,
        include=include,
        exclude=exclude,
        sort=sort_key,
        zdr=zdr,
    )
    # The listing goes out before the estimate: it is the criteria's own account of what it
    # chose, and the check results land under it once the run is confirmed.
    from provibench.bench.precheck import CheckPlan
    from provibench.bench.selection import pinned_spec, render_candidates

    for line in render_candidates(selection, model=model, sort=sort_key, zdr=zdr):
        invocation.message(line)

    candidates = [pinned_spec(gateway, model, endpoint.tag) for endpoint in selection.kept]
    specs = [*candidates, *native_specs(model, targets)]
    if not specs:
        raise InvalidInput(
            f"No endpoint of {model!r} survived the selection and no native target carries it",
            hint="Pass --include TAG to keep a dropped endpoint, or check: provibench endpoints",
        )
    criteria = Criteria(
        model=model,
        gateway=gateway,
        include=include,
        exclude=exclude,
        sort=sort_key,
        top=top,
        zdr=zdr,
        check=check or top is not None,
    )
    return execute_probe(
        invocation,
        ProbeRequest(
            trace=trace,
            specs=specs,
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
            pre_check=CheckPlan(candidates=candidates, keep=top) if criteria.check else None,
            sweep=sweep_info(selection, criteria),
        ),
    )
