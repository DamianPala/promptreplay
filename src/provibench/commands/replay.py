"""`replay`: send a trace's recorded requests to one or more targets and report cache/cost.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
functions that use them, or under `TYPE_CHECKING` for annotations, so building the CLI
(schema, --help, completion) stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import click

from provibench.commands.dry_run import DRY_RUN_OPTION, dry_run_fields
from provibench.commands.dry_run import build_document as build_dry_run_document
from provibench.commands.inspect import load_trace_entries, resolve_trace_path
from provibench.commands.prices import native_price_table
from provibench.commands.run_specs import (
    check_budget,
    endpoint_index,
    load_targets,
    parse_specs,
    select_conversation,
    spec_price_map,
    unpriced_labels,
    validate_pinned_providers,
)
from provibench.commands.summary_view import SUMMARY, render_summaries, summary_to_document
from provibench.core.confirm import require_confirmation
from provibench.core.context import Invocation
from provibench.core.documents import Document, array, boolean, integer, obj, string
from provibench.core.errors import InvalidInput, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.estimate import SpecEstimate
    from provibench.bench.replay import ReplayResult
    from provibench.bench.summary import RunSummary
    from provibench.bench.targets import RunSpec
    from provibench.bench.trace import TraceEntry

_ERROR_TRUNCATE = 120

_OUTPUT = obj(
    {
        "run_dir": string(),
        "conversation": string(),
        "turns": integer(),
        "summaries": array(SUMMARY),
        "partial": boolean(),
        "changed": boolean(),
        **dry_run_fields(),
    },
    required=["partial", "changed"],
)


@click.command(
    "replay",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.NON_IDEMPOTENT,
        confirm=True,
        output=_OUTPUT,
        output_description=(
            "partial is required: true when a turn failed against any target, false "
            "otherwise. The result document is always written to stdout, partial run "
            "included; a partial run also exits non-zero with an operation_failed error on "
            "stderr whose context carries only run_dir and run_hex. run_dir, conversation, "
            "turns, and summaries are present only on a real run; --dry-run sends nothing "
            "and instead returns estimate, total_usd, runs_dir, and requires_confirmation, "
            "with partial: false and changed: false."
        ),
        render=render_summaries,
    ),
    help="Replay a trace's recorded requests against one or more targets.\n\n"
    "TRACE is either an existing path, a name under traces_dir, or 'sample' (the packaged "
    "example). Sends every recorded turn of a conversation to each --run target with "
    "max_tokens: 1, so a run costs only prompt tokens; the output-side generation is "
    "discarded. A nonce is prepended to the first system block so the run measures its own "
    "cache rather than one left behind by an earlier run; --warm sends no nonce and reads "
    "whatever cache exists. Spends API credit, so this asks for confirmation unless --yes "
    "is given; --budget refuses a run whose worst-case cost exceeds it. The run still "
    "persists and exits non-zero when a turn failed against any target. --dry-run prices "
    "the run and stops there, sending nothing.",
)
@click.argument("trace", help="Trace path, a name under traces_dir, or 'sample'")
@click.option(
    "--run",
    "run_specs",
    multiple=True,
    required=True,
    help="One or more endpoints, `target:model[@provider[,provider...]]`; repeatable",
)
@click.option("--conversation", default=None, help="Conversation key; defaults to the main one")
@click.option("--max-tokens", type=click.IntRange(1), default=1, help="Output tokens to request")
@click.option("--delay", type=click.FloatRange(0.0), default=0.0, help="Seconds between turns")
@click.option("--strip-thinking", is_flag=True, help="Drop thinking blocks from assistant turns")
@click.option("--limit", type=click.IntRange(1), default=None, help="Replay only the first N turns")
@click.option("--warm", is_flag=True, help="Send no nonce and measure the cache as found")
@click.option(
    "--budget",
    type=click.FloatRange(0.0),
    default=None,
    help="Refuse to run when the worst-case estimate exceeds this many USD",
)
@click.option(
    "--yes", is_flag=True, help="Skip the confirmation prompt; the budget check still applies"
)
@DRY_RUN_OPTION
@click.pass_context
def replay(  # noqa: PLR0913 (click binds one parameter per flag; there is no group to extract)
    ctx: click.Context,
    *,
    trace: str,
    run_specs: tuple[str, ...],
    conversation: str | None,
    max_tokens: int,
    delay: float,
    strip_thinking: bool,
    limit: int | None,
    warm: bool,
    budget: float | None,
    yes: bool,
    dry_run: bool,
) -> Document:
    from provibench.bench.estimate import render_estimate, replay_estimate
    from provibench.bench.nonce import new_run_hex
    from provibench.bench.replay import ReplayOptions, replay_all, write_run
    from provibench.bench.summary import cache_mode_note, summarize
    from provibench.bench.trace import trace_name

    invocation = require_invocation(ctx)
    specs = parse_specs(run_specs, load_targets(invocation))
    trace_path = resolve_trace_path(trace, invocation)
    entries = load_trace_entries(trace_path)
    selected_entries, selected_key = select_conversation(entries, conversation)
    if limit is not None:
        selected_entries = selected_entries[:limit]
    if not selected_entries:
        raise InvalidInput(f"No turns to replay in conversation {selected_key!r}")

    index, lookup_notes = endpoint_index(specs)
    for note in lookup_notes:
        invocation.message(note)
    validate_pinned_providers(specs, index)
    table = native_price_table(invocation, specs)
    prices = spec_price_map(specs, index, table=table)
    estimates = [replay_estimate(spec, selected_entries, prices.get(spec.label)) for spec in specs]
    invocation.message(render_estimate(estimates))
    check_budget(estimates, budget, hint="Raise --budget, or drop a target or a turn")
    if dry_run:
        runs_dir = Path(invocation.setting("runs_dir") or ".")
        return build_dry_run_document(estimates, runs_dir=runs_dir)
    _confirm(invocation, estimates, selected_entries, specs, yes=yes)

    opts = ReplayOptions(
        max_tokens=max_tokens, delay_s=delay, strip_thinking=strip_thinking, warm=warm
    )
    run_hex = new_run_hex()
    total = len(selected_entries)

    def on_progress(result: ReplayResult) -> None:
        invocation.message(_progress_line(result, total))

    try:
        results = asyncio.run(
            replay_all(
                specs,
                selected_entries,
                opts,
                invocation.env,
                run_hex=run_hex,
                on_progress=on_progress,
                table=table,
            )
        )
    except ValueError as exc:
        raise OperationFailed(str(exc)) from exc

    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = write_run(
        runs_dir,
        trace_name(trace_path),
        selected_key,
        specs,
        opts,
        results=results,
        run_hex=run_hex,
    )
    notes = [cache_mode_note(warm)]
    summaries = [
        summarize(spec.label, results[spec.label], spec.providers, notes) for spec in specs
    ]

    document: Document = {
        "run_dir": str(run_dir),
        "conversation": selected_key,
        "turns": total,
        "summaries": [summary_to_document(s) for s in summaries],
        "partial": False,
        "changed": True,
    }
    _fail_on_partial(invocation, summaries, document, run_dir, run_hex)
    return document


def _confirm(
    invocation: Invocation,
    estimates: Sequence[SpecEstimate],
    entries: Sequence[TraceEntry],
    specs: Sequence[RunSpec],
    *,
    yes: bool,
) -> None:
    from provibench.bench.estimate import estimate_total

    total = estimate_total(estimates)
    worst = "an unknown amount" if total is None else f"up to ${total:.4f} at worst, no cache hit"
    labels = ", ".join(spec.label for spec in specs)
    question = (
        f"Replay {len(entries)} turns against {len(specs)} run(s) ({labels}), spending {worst}"
    )
    unpriced = unpriced_labels(estimates)
    if unpriced:
        question += f"; no listed price for {', '.join(unpriced)}"
    require_confirmation(invocation, question=question, yes=yes)


def _fail_on_partial(
    invocation: Invocation,
    summaries: Sequence[RunSummary],
    document: Document,
    run_dir: Path,
    run_hex: str,
) -> None:
    """Set `partial` and, when any turn failed, fail after the run is written (O5a, F1c).

    The result document reaches stdout exactly as a clean run's would, partial or not; a
    terminal instead gets the summary table on stderr before the error, which names only
    the run rather than repeating the document.
    """
    from provibench.core.output import write_document

    failed = sum(summary.errors for summary in summaries)
    document["partial"] = bool(failed)
    if not failed:
        return
    if invocation.machine_readable:
        write_document(invocation.streams.stdout, document)
    else:
        with invocation.rendering_to(invocation.streams.stderr):
            render_summaries(invocation, document)

    def hook() -> None:
        raise OperationFailed(
            f"The replay finished with {failed} failed turn(s)",
            hint=f"The run is saved; inspect it with provibench report {run_dir}",
            context={"run_dir": str(run_dir), "run_hex": run_hex},
        )

    invocation.on_success.append(hook)


def _progress_line(result: ReplayResult, total: int) -> str:
    label = (
        f"{result.model or '?'}@{','.join(result.requested_providers)}"
        if (result.requested_providers)
        else (result.model or "?")
    )
    message = (
        f"[{label}] {result.turn}/{total} {result.status} {result.provider or '-'} "
        f"prompt={result.prompt_total} cached={result.cached} {result.latency_ms:.0f}ms"
    )
    if result.error:
        message += f" error={result.error[:_ERROR_TRUNCATE]}"
    return message
