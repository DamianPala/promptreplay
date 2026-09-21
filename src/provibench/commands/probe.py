"""`probe`: measure a provider's prompt cache from a few real turns of a trace.

The command is a thin layer over `execute_probe`, which `sweep` calls too: a sweep is a
probe whose endpoints were expanded from an OpenRouter listing, and everything after that —
the estimate, the confirmation, the protocol, the run directory, the tables — is one flow.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
functions that use them, so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import click

from provibench.commands.dry_run import build_document as build_dry_run_document
from provibench.commands.dry_run import dry_run_fields
from provibench.commands.inspect import resolve_trace_path
from provibench.commands.prices import native_price_table
from provibench.commands.probe_flags import ProbeRequest, options_for, probe_options
from provibench.commands.probe_phases import (
    checked_specs,
    confirm,
    planned_specs,
    recorded_sweep,
    resolve_index,
    upper_bound_estimate,
)
from provibench.commands.probe_phases import estimates as planned_estimates
from provibench.commands.probe_result import (
    by_price,
    fail_on_partial,
    in_order,
    parallel_note,
    progress_line,
)
from provibench.commands.probe_spend import (
    LISTING_PRICES_SCHEMA,
    PRECHECK_SCHEMA,
    listing_prices_document,
    report_spend,
)
from provibench.commands.run_specs import (
    check_budget,
    load_targets,
    parse_specs,
    select_conversation,
    spec_price_map,
    validate_pinned_providers,
)
from provibench.commands.summary_view import (
    PROBE_SUMMARY,
    message_lines,
    probe_summary_to_document,
    render_probe_run,
)
from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    JsonSchema,
    array,
    boolean,
    integer,
    nullable_integer,
    nullable_number,
    obj,
    string,
)
from provibench.core.errors import InvalidInput, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.probe import ProbeResult
    from provibench.bench.targets import RunSpec

_PROBE_PROPERTIES: dict[str, JsonSchema] = {
    "run_dir": string(),
    "run_hex": string(),
    "conversation": string(),
    "rungs": array(integer()),
    "summaries": array(PROBE_SUMMARY),
    "partial": boolean(),
    "spend_usd": nullable_number(),
    "worst_case_usd": nullable_number(),
    "precheck": PRECHECK_SCHEMA,
    "trace_prompt_tokens": nullable_integer(),
    "listing_prices": LISTING_PRICES_SCHEMA,
    "changed": boolean(),
    **dry_run_fields(precheck=True),
}
# spend_usd, worst_case_usd, precheck, trace_prompt_tokens and listing_prices sit next to
# summaries: present on a real run, absent (like run_dir and summaries) on a --dry-run
# document, so only the two fields both shapes always carry are required.
_PROBE_REQUIRED = ["partial", "changed"]
_OUTPUT = obj(_PROBE_PROPERTIES, required=_PROBE_REQUIRED)


def output_fields() -> tuple[dict[str, JsonSchema], list[str]]:
    """The probe document's fields, so `sweep` can publish the same shape plus its own."""
    return dict(_PROBE_PROPERTIES), list(_PROBE_REQUIRED)


@click.command(
    "probe",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.NON_IDEMPOTENT,
        confirm=True,
        output=_OUTPUT,
        output_description=(
            "partial is required: true when a request failed or a rung was skipped, false "
            "otherwise. The result document is always written to stdout, partial run "
            "included; a partial run also exits non-zero with an operation_failed error on "
            "stderr whose context carries only run_dir and run_hex. run_dir, run_hex, "
            "conversation, rungs, and summaries are present only on a real run; --dry-run "
            "sends nothing and instead returns estimate, total_usd, pre_check, upper_bound, "
            "runs_dir, and requires_confirmation, with partial: false and changed: false."
        ),
        render=render_probe_run,
    ),
    help="Probe a trace's prompt-cache behaviour on a few of its turns.\n\n"
    "For each rung (a turn k) it sends turn k once, cold, then turn k+1 a few times to "
    "measure whether the provider cached the prefix and whether later requests found it. "
    "A fresh nonce isolates the run's cache from every earlier one, unless --warm. Every "
    "read asks for one output token, and far fewer requests than replaying the whole trace; "
    "the confirmation shows the worst-case cost of the run before anything is sent. A "
    "provider that ignores the output budget is noted in the run's own summary, not hidden "
    "from the estimate. --dry-run prices the run and stops there, sending nothing.",
)
@click.argument("trace", help="Trace path, name under traces_dir, or 'sample-trace'")
@click.argument(
    "endpoints",
    nargs=-1,
    required=True,
    help="One or more endpoints, `target:model[@provider[,provider...]]`",
)
@probe_options
@click.pass_context
def probe(  # noqa: PLR0913 (click binds one parameter per flag; there is no group to extract)
    ctx: click.Context,
    *,
    trace: str,
    endpoints: tuple[str, ...],
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
    dry_run: bool,
) -> Document:
    invocation = require_invocation(ctx)
    request = ProbeRequest(
        trace=trace,
        specs=parse_specs(endpoints, load_targets(invocation)),
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
        dry_run=dry_run,
    )
    return execute_probe(invocation, request)


def execute_probe(invocation: Invocation, request: ProbeRequest) -> Document:
    """Estimate, confirm, check availability, probe and persist one run, and return its doc.

    Nothing is sent before the confirmation: the estimate prices the availability check from
    the plan (`request.pre_check`), `--budget` is compared against that one total, and only
    a confirmed run visits the candidates.
    """
    from provibench.bench.estimate import precheck_cost, render_estimate
    from provibench.bench.probe import run_probe
    from provibench.bench.probe_drift import apply_drift
    from provibench.bench.probe_runs import endpoint_snapshot, listing_prices, write_probe_run
    from provibench.bench.probe_summary import summarize_probe
    from provibench.bench.summary import cache_mode_note
    from provibench.bench.trace import load_trace, total_prompt_tokens, trace_name

    trace_path = resolve_trace_path(request.trace, invocation)
    selected, key = select_conversation(load_trace(trace_path), request.conversation)
    if not selected:
        raise InvalidInput(f"No turns to probe in conversation {key!r}")
    options = options_for(request, selected)

    index, lookup_notes = resolve_index(request)
    for note in lookup_notes:
        invocation.message(note)
    validate_pinned_providers(request.specs, index)
    table = native_price_table(invocation, request.specs)
    prices = spec_price_map(request.specs, index, table=table)
    plan = request.pre_check
    pre_check = None if plan is None else precheck_cost(plan.candidates, selected, options, prices)
    estimates = planned_estimates(request, selected, options, prices)
    # Under `--top` the estimate prices the N best-ranked candidates while the check decides
    # which of them the run probes, so the budget is compared against what that can cost.
    upper_bound = upper_bound_estimate(request, selected, options, prices, pre_check=pre_check)
    message_lines(
        invocation, render_estimate(estimates, pre_check=pre_check, upper_bound=upper_bound)
    )
    check_budget(
        estimates,
        request.budget,
        pre_check=pre_check,
        upper_bound=upper_bound,
        hint="Raise --budget, or make the run smaller: drop an endpoint or a rung, lower "
        "--repeats, or skip the streamed request with --no-throughput",
    )
    if request.dry_run:
        runs_dir = Path(invocation.setting("runs_dir") or ".")
        return build_dry_run_document(
            estimates, runs_dir=runs_dir, pre_check=pre_check, upper_bound=upper_bound
        )
    confirm(
        invocation,
        estimates,
        options,
        planned_specs(request),
        yes=request.yes,
        pre_check=pre_check,
        upper_bound=upper_bound,
    )

    checked = checked_specs(request, selected, options, invocation)

    def on_progress(record: ProbeResult) -> None:
        invocation.message(progress_line(record))

    try:
        run = asyncio.run(
            run_probe(
                checked.specs,
                selected,
                options,
                invocation.env,
                on_progress=on_progress,
                parallel=request.parallel,
            )
        )
    except ValueError as exc:
        raise OperationFailed(str(exc)) from exc

    parallel = parallel_note(request.parallel)
    notes = [*lookup_notes, cache_mode_note(options.warm), *parallel]
    trace_prompt_tokens = total_prompt_tokens(selected)
    listing = listing_prices(checked.specs, index)
    summaries = [
        summarize_probe(
            spec.label,
            run.records[spec.label],
            prices=prices.get(spec.label),
            notes=notes,
            unpinned_gateway=spec.kind == "openrouter" and not spec.providers,
            trace_prompt_tokens=trace_prompt_tokens,
            listing=listing,
            warm=options.warm,
        )
        for spec in checked.specs
    ]
    specs: Sequence[RunSpec] = checked.specs
    if request.by_price:
        # Both the tables and the run directory take the measured order, so a later
        # `report` of this run reads it exactly as the sweep that produced it did.
        summaries = by_price(summaries)
        specs = in_order(specs, summaries)
    summaries = apply_drift(summaries, specs)

    sweep = recorded_sweep(request.sweep, checked)
    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = write_probe_run(
        runs_dir,
        # The trace's own name, so the packaged `sample-trace.jsonl.gz` files under
        # `runs/sample-trace/` exactly as `sample-trace.jsonl` would, and
        # `report sample-trace` finds it.
        trace_name(trace_path),
        key,
        run,
        specs,
        endpoints=endpoint_snapshot(specs, index),
        prices=prices,
        notes=[*lookup_notes, *parallel],
        sweep=sweep,
        precheck=checked.precheck,
        trace_prompt_tokens=trace_prompt_tokens,
        listing_prices=listing,
    )
    precheck_document, total, worst = report_spend(
        summaries,
        checked.precheck,
        prices=prices,
        records=run.records,
    )
    document: Document = {
        "run_dir": str(run_dir),
        "run_hex": run.run_hex,
        "conversation": key,
        "rungs": run.options.rungs or [],
        "summaries": [probe_summary_to_document(summary) for summary in summaries],
        "partial": False,
        "spend_usd": total,
        "worst_case_usd": worst,
        "precheck": precheck_document,
        "trace_prompt_tokens": trace_prompt_tokens,
        "listing_prices": listing_prices_document(listing),
        "changed": True,
    }
    if sweep is not None:
        # A sweep publishes the same document plus what it expanded from, so a failing
        # spec reports the model, the criteria and the drops it was part of along with
        # its numbers.
        document["sweep"] = sweep.to_document()
    fail_on_partial(invocation, run, summaries, document, run_dir)
    return document
