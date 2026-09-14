"""`history`: the runs a model already has, read as a time series.

Providers change routing, quantization, cache config and prices from week to week, so the
useful question is not "which endpoint is cheapest" once but "which one still is, and what
changed since last time". Every probe and sweep already leaves its runs under
`runs/<trace>/<timestamp>/`; this command lists them per provider and draws each spec's hit
rate across them, with no network and nothing recomputed that the run did not measure.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from pathlib import Path

import click

from provibench.commands.run_tables import render_history
from provibench.commands.run_view import HISTORY_OUTPUT, history_document
from provibench.core.documents import Document
from provibench.core.errors import NotFound
from provibench.core.params import DURATION
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects


@click.command(
    "history",
    cls=Command,
    spec=CommandSpec(effects=Effects.READ_ONLY, output=HISTORY_OUTPUT, render=render_history),
    help="List the runs of MODEL over time, one row per run and provider.\n\n"
    "MODEL is the model the runs' specs carry: the OpenRouter slug they were swept by, or "
    "the name a native target serves it under. With no MODEL, every model the runs dir "
    "holds is listed. Runs come from the configured runs dir (runs/<trace>/<timestamp>/), "
    "newest last within a provider; --trace keeps one trace's runs, --since drops the ones "
    "older than a duration (30s, 5m, 2h, 7d). Each row is what that run measured, next to "
    "the listed price its endpoint snapshot recorded at the time, and '-' where a run "
    "recorded none. The block under the table draws one sparkline per spec: its hit rate "
    "across the runs it has, oldest on the left, with the number of runs.",
)
@click.argument(
    "model",
    required=False,
    help="Model to list (deepseek/deepseek-v4.1-flash, or deepseek-flash); default: all of them",
)
@click.option("--trace", default=None, help="Only runs of this trace")
@click.option(
    "--since",
    "since_s",
    type=DURATION,
    default=None,
    metavar="DURATION",
    help="Only runs at most this old: a positive integer with s, m, h, or d",
)
@click.pass_context
def history(
    ctx: click.Context, model: str | None, trace: str | None, since_s: int | None
) -> Document:
    invocation = require_invocation(ctx)
    from provibench.bench.run_history import history_series, keep_fresh, scan_runs

    runs_dir = Path(invocation.setting("runs_dir") or ".")
    found = scan_runs(runs_dir, model=model, trace=trace)
    if not found:
        raise _nothing_found(runs_dir, model=model, trace=trace)
    # The age filter runs after the existence check, so "no run this recent" is an empty
    # table — a legitimate answer — while "no run at all" is a `not_found` with a hint.
    runs = keep_fresh(found, since_s=since_s, now=invocation.clock.now())
    return history_document(runs, history_series(runs))


def _nothing_found(runs_dir: Path, *, model: str | None, trace: str | None) -> NotFound:
    """Why nothing matched, with the names that would have: models, or traces, or neither."""
    from provibench.bench.run_history import models_in, scan_runs

    everything = scan_runs(runs_dir)
    if not everything:
        return NotFound(
            f"No runs under {runs_dir}",
            hint="Run one first: provibench probe TRACE SPEC, or provibench sweep TRACE MODEL",
        )
    if model is not None:
        known = ", ".join(models_in(everything)) or "(none)"
        return NotFound(
            f"No runs of {model!r} under {runs_dir}", hint=f"Models in the runs dir: {known}"
        )
    known = ", ".join(sorted({run.trace for run in everything})) or "(none)"
    return NotFound(
        f"No runs of trace {trace!r} under {runs_dir}", hint=f"Traces in the runs dir: {known}"
    )
