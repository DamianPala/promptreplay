"""`report`: summarise an existing run, probe or full replay in a chosen format.

The report is rendered from the same document this command prints under `--json`, so the
file and the terminal cannot disagree; `bench.html_report` does the HTML rendering.

`promptreplay.bench.*` pulls in httpx and pydantic; its symbols are imported only inside
the callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import click

from promptreplay.commands.probe_spend import (
    LISTING_PRICES_SCHEMA,
    PRECHECK_SCHEMA,
    listing_prices_document,
)
from promptreplay.commands.run_refs import resolve_run_dir
from promptreplay.commands.summary_schema import REPORTED_AVERAGE, reported_average_document
from promptreplay.commands.summary_view import (
    SWEEP_BLOCK_PROPERTIES,
    probe_summary_to_document,
    render_report_document,
    summary_to_document,
)
from promptreplay.core.context import Invocation
from promptreplay.core.documents import (
    Document,
    array,
    boolean,
    integer,
    nullable_integer,
    nullable_number,
    nullable_object,
    nullable_string,
    number,
    obj,
    string,
)
from promptreplay.core.registry import Command, require_invocation
from promptreplay.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from promptreplay.bench.replay import ReplayOptions

_OPTIONS = obj(
    {
        "max_tokens": integer(),
        "delay_s": number(),
        "strip_thinking": boolean(),
        "timeout_s": number(),
        "warm": boolean(),
        "rungs": array(integer()),
        "repeats": array(integer()),
        "gap_s": number(),
    },
    required=["strip_thinking", "timeout_s"],
)
_OUTPUT = obj(
    {
        "run_dir": string(),
        "trace": string(),
        "conversation": string(),
        "created": string(),
        "run_hex": nullable_string(),
        "protocol": string(enum=["full", "probe"]),
        "options": _OPTIONS,
        "summaries": array({}),
        "sweep": nullable_object(SWEEP_BLOCK_PROPERTIES, required=list(SWEEP_BLOCK_PROPERTIES)),
        "spend_usd": nullable_number(),
        "worst_case_usd": nullable_number(),
        "precheck": PRECHECK_SCHEMA,
        "trace_prompt_tokens": nullable_integer(),
        "listing_prices": LISTING_PRICES_SCHEMA,
        "reported_average": REPORTED_AVERAGE,
        "output_file": nullable_string(),
        "changed": boolean(),
    },
    required=[
        "run_dir",
        "trace",
        "conversation",
        "created",
        "run_hex",
        "protocol",
        "options",
        "summaries",
        "sweep",
        "spend_usd",
        "worst_case_usd",
        "precheck",
        "trace_prompt_tokens",
        "listing_prices",
        "reported_average",
        "output_file",
        "changed",
    ],
)


@click.command(
    "report",
    cls=Command,
    spec=CommandSpec(
        effects=Effects.IDEMPOTENT,
        output=_OUTPUT,
        output_description=(
            "summaries holds one object per endpoint: shaped like probe's or sweep's summaries "
            "(see promptreplay schema probe) when protocol is 'probe', or like replay's (see "
            "promptreplay schema replay) when protocol is 'full'. output_file is set when "
            "--output-file writes the selected rendering or JSON document; that file "
            "contains exactly what stdout would have received."
        ),
        render=render_report_document,
    ),
    help="Summarise an existing run.\n\n"
    "RUN is a run directory, a trace name (the newest run under <runs-dir>/<name>), or "
    "'latest' (the newest run across every trace). A probe run is summarised as hit rate, "
    "cached fraction and effective price per endpoint, a full replay as its per-turn totals. "
    "Use --format text, md, html, or json to choose the rendering. With --output-file, that "
    "rendering is written to the path, replacing what was there, and stdout stays empty; a "
    "report is derived from the run directory alone, so writing it again is the same report. "
    "For a file, md or html read best: text is the terminal table at the terminal's width.",
)
@click.argument("run", help="Run directory, trace name, or 'latest'")
@click.option(
    "--format",
    "format_name",
    type=click.Choice(["text", "md", "html", "json"]),
    default=None,
    metavar="NAME",
    help="Report rendering: text, md, html, or json (what --json selects); without it, text "
    "on a terminal and JSON otherwise",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default=None,
    metavar="PATH",
    help="Write the selected rendering to this path instead of stdout, replacing the file",
)
@click.pass_context
def report(
    ctx: click.Context,
    run: str,
    format_name: str | None,
    output_file: str | None,
) -> Document:
    invocation = require_invocation(ctx)
    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = resolve_run_dir(run, runs_dir, invocation.cwd)

    from promptreplay.bench.probe_runs import read_protocol

    output_path = _output_target(invocation) if output_file else None
    if read_protocol(run_dir) == "probe":
        document = _probe_report(run_dir)
    else:
        document = _full_report(run_dir)
    document["output_file"] = str(output_path) if output_path is not None else None
    document["changed"] = bool(document["changed"]) or output_path is not None
    return document


def _full_report(run_dir: Path) -> Document:
    from promptreplay.bench.replay import load_run
    from promptreplay.bench.summary import cache_mode_note, summarize

    meta, results = load_run(run_dir)
    notes = [cache_mode_note(meta.options.warm)]
    summaries = [
        summarize(ref.label, results[ref.label], ref.providers, notes) for ref in meta.runs
    ]
    entries = [summary_to_document(summary) for summary in summaries]
    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "run_hex": meta.run_hex,
        "protocol": meta.protocol,
        "options": _replay_options(meta.options),
        "summaries": entries,
        "sweep": None,
        "spend_usd": None,
        "worst_case_usd": None,
        "precheck": None,
        "trace_prompt_tokens": None,
        "listing_prices": [],
        "reported_average": None,
        "output_file": None,
        "changed": False,
    }


def _probe_report(run_dir: Path) -> Document:
    from promptreplay.bench.probe_drift import apply_drift
    from promptreplay.bench.probe_runs import load_probe_run
    from promptreplay.bench.probe_summary import summarize_probe
    from promptreplay.bench.reported_average import apply_reported_average
    from promptreplay.bench.spend import run_worst_case_usd, total_spend
    from promptreplay.bench.summary import cache_mode_note

    meta, records = load_probe_run(run_dir)
    notes = [*meta.notes, cache_mode_note(meta.options.warm)]
    summaries = [
        summarize_probe(
            ref.label,
            records[ref.label],
            prices=meta.prices.get(ref.label),
            notes=notes,
            unpinned_gateway=ref.kind == "openrouter" and not ref.providers,
            trace_prompt_tokens=meta.trace_prompt_tokens,
            listing=meta.listing_prices,
            warm=meta.options.warm,
        )
        for ref in meta.specs
    ]
    specs = meta.specs
    if meta.sweep is not None:
        # A sweep's order is its measured price. The run directory stores the order the
        # sweep printed, but the price is recomputed here from the records, so a pricing
        # rule newer than the run would otherwise leave a stale order under a fresh table.
        from promptreplay.commands.probe_result import by_price

        summaries = by_price(summaries)
        by_label = {ref.label: ref for ref in meta.specs}
        specs = [by_label[summary.label] for summary in summaries]
    summaries = apply_drift(summaries, specs)
    summaries = apply_reported_average(
        summaries, specs, meta.reported_average, trace_prompt_tokens=meta.trace_prompt_tokens
    )
    spend = total_spend([summary.spend_usd for summary in summaries])
    worst = run_worst_case_usd(
        records,
        meta.prices,
        precheck_worst_case_usd=meta.precheck.worst_case_usd if meta.precheck else None,
    )
    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "run_hex": meta.run_hex,
        "protocol": meta.protocol,
        "options": dict(meta.options.model_dump()),
        "summaries": [probe_summary_to_document(s) for s in summaries],
        # The selection the run chose its endpoints by, so the report can print it with no
        # network; `None` for a probe whose endpoints were given by hand.
        "sweep": None if meta.sweep is None else meta.sweep.to_document(),
        "spend_usd": total_spend([spend, meta.precheck.spend_usd if meta.precheck else None]),
        "worst_case_usd": worst,
        "precheck": meta.precheck.model_dump() if meta.precheck is not None else None,
        "trace_prompt_tokens": meta.trace_prompt_tokens,
        "listing_prices": listing_prices_document(meta.listing_prices),
        "reported_average": reported_average_document(meta.reported_average),
        "output_file": None,
        "changed": False,
    }


def _replay_options(options: "ReplayOptions") -> Document:
    """The full-replay options as a document, without the probe-only fields."""
    return {
        "max_tokens": options.max_tokens,
        "delay_s": options.delay_s,
        "strip_thinking": options.strip_thinking,
        "timeout_s": options.timeout_s,
        "warm": options.warm,
    }


def _output_target(invocation: Invocation) -> Path:
    """Resolve the report result path; an existing file is replaced.

    A report is a function of the run directory alone, so a repeat writes the same bytes and
    the command can keep its `idempotent` claim (R1) without a `--force` gate.
    """
    target = invocation.output_file
    if target is None:
        raise RuntimeError("--output-file was not resolved before report callback")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
