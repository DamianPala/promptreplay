"""`replay`: send a trace's recorded requests to one or more targets and report cache/cost.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
functions that use them, or under `TYPE_CHECKING` for annotations, so building the CLI
(schema, --help, completion) stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import click

from provibench.commands.inspect import resolve_trace_path
from provibench.commands.summary_view import SUMMARY, render_summaries, summary_to_document
from provibench.core.confirm import require_confirmation
from provibench.core.context import Invocation
from provibench.core.documents import Document, array, boolean, integer, obj, string
from provibench.core.errors import InvalidInput, NotFound, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.replay import ReplayResult
    from provibench.bench.targets import RunSpec, Target
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
    "Sends every recorded turn of a conversation to each --run target with "
    "max_tokens: 1, so a run costs only prompt tokens; the output-side generation is "
    "discarded. Spends API credit, so this asks for confirmation unless --yes is given.",
)
@click.argument("trace")
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
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt")
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
    yes: bool,
) -> Document:
    from provibench.bench.replay import ReplayOptions, replay_all, write_run
    from provibench.bench.summary import summarize
    from provibench.bench.trace import load_trace

    invocation = require_invocation(ctx)
    targets = _load_targets(invocation)
    specs = _parse_specs(run_specs, targets)

    trace_path = resolve_trace_path(trace, invocation)
    entries = load_trace(trace_path)
    selected_entries, selected_key = _select_conversation(entries, conversation)
    if limit is not None:
        selected_entries = selected_entries[:limit]
    if not selected_entries:
        raise InvalidInput(f"No turns to replay in conversation {selected_key!r}")

    _confirm(invocation, selected_entries, specs, yes=yes)

    opts = ReplayOptions(max_tokens=max_tokens, delay_s=delay, strip_thinking=strip_thinking)
    total = len(selected_entries)

    def on_progress(result: ReplayResult) -> None:
        invocation.message(_progress_line(result, total))

    try:
        results = asyncio.run(
            replay_all(specs, selected_entries, opts, invocation.env, on_progress=on_progress)
        )
    except ValueError as exc:
        raise OperationFailed(str(exc)) from exc

    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = write_run(runs_dir, trace_path.stem, selected_key, specs, opts, results=results)
    summaries = [summarize(spec.label, results[spec.label], spec.providers) for spec in specs]

    return {
        "run_dir": str(run_dir),
        "conversation": selected_key,
        "turns": total,
        "summaries": [summary_to_document(s) for s in summaries],
        "changed": True,
    }


def _load_targets(invocation: Invocation) -> dict[str, Target]:
    targets_path = Path(invocation.setting("targets_path") or "")
    if targets_path.is_file():
        return _parse_targets_file(targets_path)
    invocation.message(f"No targets file at {targets_path}; using the packaged defaults")
    packaged = resources.files("provibench.data").joinpath("targets.toml")
    with resources.as_file(packaged) as path:
        return _parse_targets_file(path)


def _parse_targets_file(path: Path) -> dict[str, Target]:
    from provibench.bench.targets import load_targets

    try:
        return load_targets(path)
    except ValueError as exc:
        raise InvalidInput(str(exc)) from exc


def _parse_specs(run_specs: Iterable[str], targets: dict[str, Target]) -> list[RunSpec]:
    from provibench.bench.targets import parse_run_spec

    specs: list[RunSpec] = []
    for raw in run_specs:
        try:
            specs.append(parse_run_spec(raw, targets))
        except ValueError as exc:
            raise InvalidInput(str(exc)) from exc
    return specs


def _select_conversation(
    entries: list[TraceEntry], conversation: str | None
) -> tuple[list[TraceEntry], str]:
    from provibench.bench.trace import group_conversations, main_conversation

    groups = group_conversations(entries)
    if conversation is not None:
        if conversation not in groups:
            raise NotFound(f"No conversation {conversation!r} in this trace")
        return sorted(groups[conversation], key=lambda e: e.seq), conversation
    main = main_conversation(entries)
    if not main:
        raise InvalidInput("This trace has no conversations to replay")
    return sorted(main, key=lambda e: e.seq), main[0].conversation


def _confirm(
    invocation: Invocation, entries: list[TraceEntry], specs: list[RunSpec], *, yes: bool
) -> None:
    prompt_total = sum(e.response.usage.prompt_total for e in entries if e.response)
    labels = ", ".join(spec.label for spec in specs)
    question = (
        f"Replay {len(entries)} turns (~{prompt_total} recorded prompt tokens) "
        f"against {len(specs)} run(s): {labels}"
    )
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
