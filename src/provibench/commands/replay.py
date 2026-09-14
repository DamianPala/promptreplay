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

from provibench.commands.inspect import load_trace_entries, resolve_trace_path
from provibench.commands.run_specs import (
    check_budget,
    endpoint_index,
    load_targets,
    parse_specs,
    select_conversation,
    spec_price_map,
    unpriced_labels,
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
    from provibench.bench.targets import RunSpec
    from provibench.bench.trace import TraceEntry

_ERROR_TRUNCATE = 120

_OUTPUT = obj(
    {
        "run_dir": string(),
        "conversation": string(),
        "turns": integer(),
        "summaries": array(SUMMARY),
        "changed": boolean(),
    },
    required=["run_dir", "conversation", "turns", "summaries", "changed"],
)


@click.command(
    "replay",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.NON_IDEMPOTENT, confirm=True, output=_OUTPUT, render=render_summaries
    ),
    help="Replay a trace's recorded requests against one or more targets.\n\n"
    "TRACE is either an existing path, a name under traces_dir, or 'sample' (the packaged "
    "example). Sends every recorded turn of a conversation to each --run target with "
    "max_tokens: 1, so a run costs only prompt tokens; the output-side generation is "
    "discarded. A nonce is prepended to the first system block so the run measures its own "
    "cache rather than one left behind by an earlier run; --warm sends no nonce and reads "
    "whatever cache exists. Spends API credit, so this asks for confirmation unless --yes "
    "is given; --budget refuses a run whose worst-case cost exceeds it.",
)
@click.argument("trace", help="Trace path, a name under traces_dir, or 'sample'")
@click.option(
    "--run",
    "run_specs",
    multiple=True,
    required=True,
    help="Run spec <target>:<model>[@provider[,provider]]; repeatable",
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
    prices = spec_price_map(specs, index)
    estimates = [replay_estimate(spec, selected_entries, prices.get(spec.label)) for spec in specs]
    invocation.message(render_estimate(estimates))
    check_budget(estimates, budget, hint="Raise --budget, or drop a target or a turn")
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

    return {
        "run_dir": str(run_dir),
        "conversation": selected_key,
        "turns": total,
        "summaries": [summary_to_document(s) for s in summaries],
        "changed": True,
    }


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
