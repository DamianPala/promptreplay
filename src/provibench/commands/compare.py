"""`compare`: two runs of one trace, subtracted spec by spec.

The question `history` raises — what changed since last time — is answered by one row per
spec with the metric on either side and the difference between them. Both runs go back
through their own loader and the same summarising the report does, so a delta is two
numbers `report` would print, subtracted once.

Two runs of different traces or different protocols are refused: a full replay's per-turn
totals and a probe's warm-read hit rate are not the same measurement, and two traces are
not the same payload. `--force` compares them anyway, for a reader who knows why.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from pathlib import Path

import click

from provibench.commands.run_refs import resolve_run_dir
from provibench.commands.run_tables import render_compare
from provibench.commands.run_view import COMPARE_OUTPUT, compare_document
from provibench.core.documents import Document
from provibench.core.errors import InvalidInput
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects


@click.command(
    "compare",
    cls=Command,
    spec=CommandSpec(effects=Effects.READ_ONLY, output=COMPARE_OUTPUT, render=render_compare),
    help="Compare two runs of one trace, spec by spec.\n\n"
    "RUN_A and RUN_B each name a run directory, a trace (its newest run), or latest and "
    "previous — the two newest runs of --trace, or of the trace the newest run belongs to. "
    "Every spec both runs measured gets a row with the metric as A -> B and the change "
    "between them; the listed prices are compared from the endpoint snapshot each run "
    "recorded, where both have one; a spec only one run measured is named under the table. "
    "Runs of different traces or protocols are refused unless --force, because their "
    "numbers are not the same measurement.",
)
@click.argument("run_a", help="The earlier run: a directory, a trace name, or latest/previous")
@click.argument("run_b", help="The later run, named the same ways")
@click.option(
    "--trace",
    default=None,
    help="Trace that latest and previous name the two newest runs of; default: the one the "
    "newest run belongs to",
)
@click.option(
    "--force",
    is_flag=True,
    help="Compare runs of different traces or protocols anyway",
)
@click.pass_context
def compare(ctx: click.Context, run_a: str, run_b: str, trace: str | None, force: bool) -> Document:
    invocation = require_invocation(ctx)
    from provibench.bench.run_delta import compare_runs
    from provibench.bench.run_history import read_run

    runs_dir = Path(invocation.setting("runs_dir") or ".")
    earlier = read_run(resolve_run_dir(run_a, runs_dir, invocation.cwd, trace=trace, previous=True))
    later = read_run(resolve_run_dir(run_b, runs_dir, invocation.cwd, trace=trace, previous=True))
    if not force:
        _require_comparable(earlier.trace, later.trace, earlier.protocol, later.protocol)
    return compare_document(compare_runs(earlier, later))


def _require_comparable(trace_a: str, trace_b: str, protocol_a: str, protocol_b: str) -> None:
    """Refuse a pair whose deltas would mean nothing, naming what differs."""
    if trace_a != trace_b:
        raise InvalidInput(
            f"The two runs are of different traces: {trace_a!r} and {trace_b!r}",
            hint="Compare two runs of one trace, or pass --force to compare across traces",
        )
    if protocol_a != protocol_b:
        raise InvalidInput(
            f"The two runs used different protocols: {protocol_a!r} and {protocol_b!r}",
            hint="Compare two probe runs, or two full replays; --force compares them anyway",
        )
