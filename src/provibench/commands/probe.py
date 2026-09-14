"""`probe`: measure a provider's prompt cache from a few real turns of a trace.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import click

from provibench.commands.inspect import resolve_trace_path
from provibench.commands.run_specs import (
    check_budget,
    endpoint_index,
    load_targets,
    parse_specs,
    select_conversation,
    spec_price_map,
    unpriced_labels,
)
from provibench.commands.summary_view import (
    PROBE_SUMMARY,
    probe_summary_to_document,
    probe_tables_text,
    render_probe_run,
)
from provibench.core.confirm import require_confirmation
from provibench.core.context import Invocation
from provibench.core.documents import Document, array, boolean, integer, obj, string
from provibench.core.errors import InvalidInput, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.estimate import SpecEstimate
    from provibench.bench.probe import ProbeOptions, ProbeResult, ProbeRun
    from provibench.bench.probe_summary import ProbeSummary
    from provibench.bench.targets import RunSpec
    from provibench.bench.trace import TraceEntry

_ERROR_TRUNCATE = 120

_OUTPUT = obj(
    {
        "run_dir": string(),
        "run_hex": string(),
        "conversation": string(),
        "rungs": array(integer()),
        "summaries": array(PROBE_SUMMARY),
        "changed": boolean(),
    },
    required=["run_dir", "run_hex", "conversation", "rungs", "summaries", "changed"],
)


@click.command(
    "probe",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.NON_IDEMPOTENT, confirm=True, output=_OUTPUT, render=render_probe_run
    ),
    help="Probe a trace's prompt-cache behaviour on a few of its turns.\n\n"
    "For each rung (a turn k) it sends turn k once, cold, then turn k+1 a few times to "
    "measure whether the provider cached the prefix and whether later requests found it. "
    "A fresh nonce isolates the run's cache from every earlier one, unless --warm. Costs "
    "prompt tokens only, and far fewer than replaying the whole trace; the confirmation "
    "shows the worst-case cost of the run before anything is sent.",
)
@click.argument("trace")
@click.argument("specs", nargs=-1, required=True)
@click.option(
    "--rungs",
    default=None,
    help="1-based turn indices to probe, comma-separated; each needs a following turn. "
    "Defaults to the smallest, middle and largest turn of the conversation",
)
@click.option(
    "--repeats",
    default="6,2,2",
    show_default=True,
    help="Warm reads per rung, comma-separated; the last value broadcasts to later rungs",
)
@click.option(
    "--gap",
    type=click.FloatRange(0.0),
    default=1.0,
    show_default=True,
    help="Seconds between warm reads",
)
@click.option("--warm", is_flag=True, help="Send no nonce and measure the cache as found")
@click.option(
    "--ttl",
    default=None,
    help="Second offsets after the first served rung's warm reads to re-read its cache at, "
    "comma-separated and ascending (e.g. 60,300,900); off by default, it costs wall time",
)
@click.option(
    "--no-throughput",
    is_flag=True,
    help="Skip each rung's streamed generation request, so no TTFT, tok/s or fingerprint",
)
@click.option("--conversation", default=None, help="Conversation key; defaults to the main one")
@click.option("--strip-thinking", is_flag=True, help="Drop thinking blocks from assistant turns")
@click.option(
    "--timeout",
    type=click.FloatRange(0.1),
    default=300.0,
    show_default=True,
    help="Request timeout, seconds",
)
@click.option(
    "--budget",
    type=click.FloatRange(0.0),
    default=None,
    help="Refuse to run when the worst-case estimate exceeds this many USD",
)
@click.option(
    "--yes", is_flag=True, help="Skip the confirmation prompt; the budget check still applies"
)
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
) -> Document:
    from provibench.bench.estimate import probe_estimate, render_estimate
    from provibench.bench.probe import run_probe
    from provibench.bench.probe_drift import apply_drift
    from provibench.bench.probe_runs import write_probe_run
    from provibench.bench.probe_summary import summarize_probe
    from provibench.bench.summary import cache_mode_note
    from provibench.bench.trace import load_trace

    invocation = require_invocation(ctx)
    run_specs = parse_specs(specs, load_targets(invocation))
    trace_path = resolve_trace_path(trace, invocation)
    selected, key = select_conversation(load_trace(trace_path), conversation)
    if not selected:
        raise InvalidInput(f"No turns to probe in conversation {key!r}")
    options = _options(
        selected,
        rungs=rungs,
        repeats=repeats,
        gap=gap,
        warm=warm,
        ttl=ttl,
        throughput=not no_throughput,
        strip_thinking=strip_thinking,
        timeout=timeout,
    )

    index, lookup_notes = endpoint_index(run_specs)
    for note in lookup_notes:
        invocation.message(note)
    prices = spec_price_map(run_specs, index)
    estimates = [
        probe_estimate(spec, selected, options, prices.get(spec.label)) for spec in run_specs
    ]
    invocation.message(render_estimate(estimates))
    check_budget(estimates, budget, hint="Raise --budget, or drop a spec or a rung")
    _confirm(invocation, estimates, options, run_specs, yes=yes)

    def on_progress(record: ProbeResult) -> None:
        invocation.message(_progress_line(record))

    try:
        run = asyncio.run(
            run_probe(run_specs, selected, options, invocation.env, on_progress=on_progress)
        )
    except ValueError as exc:
        raise OperationFailed(str(exc)) from exc

    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = write_probe_run(
        runs_dir,
        trace_path.stem,
        key,
        run,
        run_specs,
        endpoints=index,
        prices=prices,
        notes=lookup_notes,
    )
    notes = [*lookup_notes, cache_mode_note(options.warm)]
    summaries = apply_drift(
        [
            summarize_probe(
                spec.label, run.records[spec.label], prices=prices.get(spec.label), notes=notes
            )
            for spec in run_specs
        ],
        run_specs,
    )
    document: Document = {
        "run_dir": str(run_dir),
        "run_hex": run.run_hex,
        "conversation": key,
        "rungs": run.options.rungs or [],
        "summaries": [probe_summary_to_document(summary) for summary in summaries],
        "changed": True,
    }
    _fail_on_partial(invocation, run, summaries, document, run_dir)
    return document


def _options(  # noqa: PLR0913 (one keyword per flag the command binds)
    conversation: Sequence[TraceEntry],
    *,
    rungs: str | None,
    repeats: str,
    gap: float,
    warm: bool,
    ttl: str | None,
    throughput: bool,
    strip_thinking: bool,
    timeout: float,
) -> ProbeOptions:
    """The run's options, with rungs and repeats resolved against the conversation."""
    from pydantic import ValidationError

    from provibench.bench.probe import ProbeOptions
    from provibench.bench.rungs import parse_int_list

    try:
        parsed_rungs = None if rungs is None else parse_int_list(rungs)
        parsed_repeats = parse_int_list(repeats)
        parsed_ttl = None if ttl is None else parse_int_list(ttl)
        options = ProbeOptions(
            rungs=parsed_rungs,
            repeats=parsed_repeats,
            gap_s=gap,
            warm=warm,
            throughput=throughput,
            ttl_s=parsed_ttl,
            strip_thinking=strip_thinking,
            timeout_s=timeout,
        )
        return options.resolved(conversation)
    except (ValueError, ValidationError) as exc:
        # A bad list, a `--ttl` offset the `--gap` already passed, an impossible rung: all
        # of them are input errors, and all of them are caught before a request is sent.
        raise InvalidInput(_first_detail(exc)) from exc


def _first_detail(error: Exception) -> str:
    """One line for a validation error, so the message reads like the other input errors."""
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        return str(error.errors()[0]["msg"])
    return str(error)


def _requests(options: ProbeOptions) -> int:
    """How many requests one spec sends: cold and warm per rung, plus the extras."""
    per_spec = sum(1 + count for count in options.repeats)
    if options.throughput:
        per_spec += len(options.repeats)
    if options.ttl_s:
        per_spec += len(options.ttl_s)
    return per_spec


def _confirm(
    invocation: Invocation,
    estimates: Sequence[SpecEstimate],
    options: ProbeOptions,
    specs: Sequence[RunSpec],
    *,
    yes: bool,
) -> None:
    from provibench.bench.estimate import estimate_total

    total = estimate_total(estimates)
    worst = "an unknown amount" if total is None else f"up to ${total:.4f} at worst, no cache hit"
    requests = len(specs) * _requests(options)
    question = (
        f"Send {requests} probe request(s) to {len(specs)} spec(s) over turn(s) "
        f"{options.rungs}, spending {worst}"
    )
    if options.ttl_s:
        question += f"; TTL re-reads at {options.ttl_s}s on the first rung"
    unpriced = unpriced_labels(estimates)
    if unpriced:
        question += f"; no listed price for {', '.join(unpriced)}"
    require_confirmation(invocation, question=question, yes=yes)


def _fail_on_partial(
    invocation: Invocation,
    run: ProbeRun,
    summaries: Sequence[ProbeSummary],
    document: Document,
    run_dir: Path,
) -> None:
    """Fail the command, after the run is persisted, when anything failed or was skipped.

    The run is worth money, so the numbers survive the failure: a terminal gets the two
    tables on stderr before the error lands, and a machine-readable call gets the same
    document it would have printed, under the error's structured `context`.
    """
    from provibench.bench.probe import is_failed

    failed = sum(1 for records in run.records.values() for record in records if is_failed(record))
    skipped = sum(summary.skipped for summary in summaries)
    if not failed and not skipped:
        return
    parts: list[str] = []
    if failed:
        parts.append(f"{failed} request(s) failed")
    if skipped:
        parts.append(f"{skipped} rung(s) skipped")
    context: Document = {"run_dir": str(run_dir), "run_hex": run.run_hex}
    if invocation.machine_readable:
        context = dict(document)
    else:
        # A multi-line message would be escaped into one line, so each line goes separately.
        for line in probe_tables_text(document).splitlines():
            invocation.message(line)

    def hook() -> None:
        raise OperationFailed(
            f"The probe finished with {' and '.join(parts)}",
            hint=f"The run is saved; inspect it with provibench report {run_dir}",
            context=context,
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
