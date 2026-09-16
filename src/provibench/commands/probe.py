"""`probe`: measure a provider's prompt cache from a few real turns of a trace.

The command is a thin layer over `execute_probe`, which `sweep` calls too: a sweep is a
probe whose specs were expanded from an endpoint list, and everything after that — the
estimate, the confirmation, the protocol, the run directory, the tables — is one flow.

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
    Checked,
    confirm,
    planned_specs,
    recorded_sweep,
    run_check,
    upper_bound_estimate,
)
from provibench.commands.probe_phases import estimates as planned_estimates
from provibench.commands.run_specs import (
    check_budget,
    endpoint_index,
    load_targets,
    parse_specs,
    select_conversation,
    spec_price_map,
    validate_pinned_providers,
)
from provibench.commands.summary_view import (
    PROBE_SUMMARY,
    message_lines,
    probe_report_text,
    probe_summary_to_document,
    render_probe_run,
)
from provibench.core.context import Invocation
from provibench.core.documents import Document, JsonSchema, array, boolean, integer, obj, string
from provibench.core.errors import InvalidInput, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint
    from provibench.bench.probe import ProbeOptions, ProbeResult, ProbeRun
    from provibench.bench.probe_summary import ProbeSummary
    from provibench.bench.targets import RunSpec
    from provibench.bench.trace import TraceEntry

_ERROR_TRUNCATE = 120

_PROBE_PROPERTIES: dict[str, JsonSchema] = {
    "run_dir": string(),
    "run_hex": string(),
    "conversation": string(),
    "rungs": array(integer()),
    "summaries": array(PROBE_SUMMARY),
    "partial": boolean(),
    "changed": boolean(),
    **dry_run_fields(precheck=True),
}
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
    "A fresh nonce isolates the run's cache from every earlier one, unless --warm. Costs "
    "prompt tokens only, and far fewer than replaying the whole trace; the confirmation "
    "shows the worst-case cost of the run before anything is sent. --dry-run prices the run "
    "and stops there, sending nothing.",
)
@click.argument("trace", help="Trace path, name under traces_dir, or 'sample'")
@click.argument("specs", nargs=-1, required=True, help="One or more target:model[@provider] specs")
@probe_options
@click.pass_context
def probe(  # noqa: PLR0913 (click binds one parameter per flag; there is no group to extract)
    ctx: click.Context,
    *,
    trace: str,
    specs: tuple[str, ...],
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
        specs=parse_specs(specs, load_targets(invocation)),
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
    from provibench.bench.probe_runs import endpoint_snapshot, write_probe_run
    from provibench.bench.probe_summary import summarize_probe
    from provibench.bench.summary import cache_mode_note
    from provibench.bench.trace import load_trace, trace_name

    trace_path = resolve_trace_path(request.trace, invocation)
    selected, key = select_conversation(load_trace(trace_path), request.conversation)
    if not selected:
        raise InvalidInput(f"No turns to probe in conversation {key!r}")
    options = options_for(request, selected)

    index, lookup_notes = _resolve_index(request)
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
        hint="Raise --budget, or make the run smaller: drop a spec or a rung, lower "
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

    checked = _checked(request, selected, options, invocation)

    def on_progress(record: ProbeResult) -> None:
        invocation.message(_progress_line(record))

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

    parallel = _parallel_note(request.parallel)
    notes = [*lookup_notes, cache_mode_note(options.warm), *parallel]
    summaries = [
        summarize_probe(
            spec.label, run.records[spec.label], prices=prices.get(spec.label), notes=notes
        )
        for spec in checked.specs
    ]
    specs: Sequence[RunSpec] = checked.specs
    if request.by_price:
        # Both the tables and the run directory take the measured order, so a later
        # `report` of this run reads it exactly as the sweep that produced it did.
        summaries = _by_price(summaries)
        specs = _in_order(specs, summaries)
    summaries = apply_drift(summaries, specs)

    sweep = recorded_sweep(request.sweep, checked)
    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = write_probe_run(
        runs_dir,
        # The trace's own name, so the packaged `sample.jsonl.gz` files under `sample/`
        # exactly as `sample.jsonl` would, and `report sample` finds it.
        trace_name(trace_path),
        key,
        run,
        specs,
        endpoints=endpoint_snapshot(specs, index),
        prices=prices,
        notes=[*lookup_notes, *parallel],
        sweep=sweep,
    )
    document: Document = {
        "run_dir": str(run_dir),
        "run_hex": run.run_hex,
        "conversation": key,
        "rungs": run.options.rungs or [],
        "summaries": [probe_summary_to_document(summary) for summary in summaries],
        "partial": False,
        "changed": True,
    }
    if sweep is not None:
        # A sweep publishes the same document plus what it expanded from, so a failing
        # spec reports the model, the criteria and the drops it was part of along with
        # its numbers.
        document["sweep"] = sweep.to_document()
    _fail_on_partial(invocation, run, summaries, document, run_dir)
    return document


def _resolve_index(request: ProbeRequest) -> tuple[dict[str, list[Endpoint]], list[str]]:
    """The endpoint index a sweep already fetched, or a fresh lookup for a plain probe."""
    if request.endpoints is None:
        return endpoint_index(request.specs)
    return dict(request.endpoints), []


def _checked(
    request: ProbeRequest,
    selected: Sequence[TraceEntry],
    options: ProbeOptions,
    invocation: Invocation,
) -> Checked:
    """The availability phase, or every spec when the run planned no check."""
    if request.pre_check is None:
        return Checked(list(request.specs), [])
    return run_check(invocation, request.pre_check, request.specs, selected, options)


def _parallel_note(parallel: int) -> list[str]:
    """One note for a run that probed several specs at once, and none for a sequential one.

    A `run.json` reader comparing latency medians between runs has to know which of them
    were measured with other specs in flight on the same connection pool; the note is part
    of the run's record, not of its options, because it describes how the numbers were
    taken rather than what the protocol did.
    """
    if parallel <= 1:
        return []
    return [f"parallel {parallel}: latency measured with specs in flight together"]


def _by_price(summaries: Sequence[ProbeSummary]) -> list[ProbeSummary]:
    """The summaries as a sweep reads them: cheapest effective prompt token first.

    That is the question a sweep asks — which endpoint to use this week — so the effective
    price leads and the hit rate breaks its ties, being the other half of why one endpoint
    is cheaper than another. A spec with no listed price cannot be ranked and sorts last.
    """
    return sorted(
        summaries,
        key=lambda s: (s.eff_per_m_prompt is None, s.eff_per_m_prompt or 0.0, -(s.hit_rate or 0.0)),
    )


def _in_order(specs: Sequence[RunSpec], summaries: Sequence[ProbeSummary]) -> list[RunSpec]:
    """The specs in the order their summaries came out, for the run directory."""
    by_label = {spec.label: spec for spec in specs}
    return [by_label[summary.label] for summary in summaries]


def _fail_on_partial(
    invocation: Invocation,
    run: ProbeRun,
    summaries: Sequence[ProbeSummary],
    document: Document,
    run_dir: Path,
) -> None:
    """Set `partial` and, when anything failed or was skipped, fail after the run is written.

    O5a: the result document is emitted on stdout exactly as a clean run's would be, partial
    or not, so an agent never has to unwrap it from `error.context`; a terminal instead gets
    the two tables on stderr before the error, which stays this command's own record and
    names only the run rather than repeating the document.
    """
    from provibench.bench.probe import is_failed
    from provibench.core.output import write_document

    failed = sum(1 for records in run.records.values() for record in records if is_failed(record))
    skipped = sum(summary.skipped for summary in summaries)
    document["partial"] = bool(failed or skipped)
    if not failed and not skipped:
        return
    parts: list[str] = []
    if failed:
        parts.append(f"{failed} request(s) failed")
    if skipped:
        parts.append(f"{skipped} rung(s) skipped")
    if invocation.machine_readable:
        write_document(invocation.streams.stdout, document)
    else:
        message_lines(invocation, probe_report_text(document))

    def hook() -> None:
        raise OperationFailed(
            f"The probe finished with {' and '.join(parts)}",
            hint=f"The run is saved; inspect it with provibench report {run_dir}",
            context={"run_dir": str(run_dir), "run_hex": run.run_hex},
        )

    invocation.on_success.append(hook)


def _progress_line(record: ProbeResult) -> str:
    """One line per request; a TTL read names its offset, a stream names its TTFT."""
    what = f"ttl {record.attempt}s" if record.role == "ttl" else f"{record.role} {record.attempt}"
    message = (
        f"{record.spec_label} rung {record.rung} {what} "
        f"{record.status} {record.cached}/{record.prompt_total} {record.latency_ms:.0f}ms"
    )
    if record.ttft_ms is not None:
        message += f" ttft={record.ttft_ms:.0f}ms"
    if record.error:
        message += f" error={record.error[:_ERROR_TRUNCATE]}"
    return message
