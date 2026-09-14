"""`report`: summarise an existing run, probe or full replay, as text, markdown or HTML.

The HTML is rendered from the same document this command prints under `--json`, so the
file and the terminal cannot disagree; `bench.html_report` does the rendering and this
module only decides where the file goes.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside
the callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import click

from provibench.commands.summary_view import (
    PROBE_SUMMARY,
    SUMMARY,
    probe_summary_to_document,
    render_report_document,
    summary_to_document,
)
from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    array,
    boolean,
    integer,
    nullable_string,
    number,
    obj,
    string,
)
from provibench.core.errors import NotFound, PreconditionFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects

if TYPE_CHECKING:
    from provibench.bench.replay import ReplayOptions

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
        "summaries": array(SUMMARY),
        "probe_summaries": array(PROBE_SUMMARY),
        "markdown_path": nullable_string(),
        "html": nullable_string(),
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
        "probe_summaries",
        "markdown_path",
        "html",
        "changed",
    ],
)


@click.command(
    "report",
    cls=Command,
    spec=CommandSpec(effects=Effects.IDEMPOTENT, output=_OUTPUT, render=render_report_document),
    help="Summarise an existing run.\n\n"
    "RUN is a run directory, a trace name (the newest run under <runs-dir>/<name>), or "
    "'latest' (the newest run across every trace). A probe run is summarised as hit rate, "
    "prefix fraction and effective price per spec, a full replay as its per-turn totals. "
    "With --md, the same report is also written as markdown; with --html, as one "
    "self-contained HTML file with the same tables and a chart.",
)
@click.argument("run")
@click.option("--md", "md_path", default=None, help="Write the report as markdown to this path")
@click.option(
    "--html",
    "html_path",
    default=None,
    metavar="PATH",
    help="Write the report as one self-contained HTML file to this path",
)
@click.option("--force", is_flag=True, help="Overwrite an existing --md or --html file")
@click.pass_context
def report(
    ctx: click.Context, run: str, md_path: str | None, html_path: str | None, force: bool
) -> Document:
    invocation = require_invocation(ctx)
    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = _resolve_run_dir(run, runs_dir, invocation.cwd)

    from provibench.bench.probe_runs import read_protocol

    # Both targets are refused before either is written: the markdown goes out while the
    # document is built, so checking the HTML only at the end would leave one of the two
    # files behind on a command that failed.
    if html_path is not None:
        _output_target(invocation, html_path, force, option="html")
    if read_protocol(run_dir) == "probe":
        document = _probe_report(invocation, run_dir, md_path, force)
    else:
        document = _full_report(invocation, run_dir, md_path, force)
    document["html"] = _write_html(invocation, html_path, force, document)
    document["changed"] = bool(document["changed"]) or document["html"] is not None
    return document


def _full_report(
    invocation: Invocation, run_dir: Path, md_path: str | None, force: bool
) -> Document:
    from provibench.bench.replay import load_run
    from provibench.bench.summary import cache_mode_note, render_markdown, summarize

    meta, results = load_run(run_dir)
    notes = [cache_mode_note(meta.options.warm)]
    summaries = [
        summarize(ref.label, results[ref.label], ref.providers, notes) for ref in meta.runs
    ]
    entries = [summary_to_document(summary) for summary in summaries]
    markdown_path = _write_markdown(invocation, md_path, render_markdown(entries), force)
    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "run_hex": meta.run_hex,
        "protocol": meta.protocol,
        "options": _replay_options(meta.options),
        "summaries": entries,
        "probe_summaries": [],
        "markdown_path": markdown_path,
        "html": None,
        "changed": markdown_path is not None,
    }


def _probe_report(
    invocation: Invocation, run_dir: Path, md_path: str | None, force: bool
) -> Document:
    from provibench.bench.probe_drift import apply_drift
    from provibench.bench.probe_runs import load_probe_run
    from provibench.bench.probe_summary import summarize_probe
    from provibench.bench.probe_tables import probe_markdown
    from provibench.bench.summary import cache_mode_note

    meta, records = load_probe_run(run_dir)
    notes = [*meta.notes, cache_mode_note(meta.options.warm)]
    summaries = apply_drift(
        [
            summarize_probe(
                ref.label, records[ref.label], prices=meta.prices.get(ref.label), notes=notes
            )
            for ref in meta.specs
        ],
        meta.specs,
    )
    markdown_path = _write_markdown(invocation, md_path, probe_markdown(summaries), force)
    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "run_hex": meta.run_hex,
        "protocol": meta.protocol,
        "options": dict(meta.options.model_dump()),
        "summaries": [],
        "probe_summaries": [probe_summary_to_document(s) for s in summaries],
        "markdown_path": markdown_path,
        "html": None,
        "changed": markdown_path is not None,
    }


def _replay_options(options: ReplayOptions) -> Document:
    """The full-replay options as a document, without the probe-only fields."""
    return {
        "max_tokens": options.max_tokens,
        "delay_s": options.delay_s,
        "strip_thinking": options.strip_thinking,
        "timeout_s": options.timeout_s,
        "warm": options.warm,
    }


def _write_html(
    invocation: Invocation, html_path: str | None, force: bool, document: Document
) -> str | None:
    """Render the report as one HTML file, refusing to replace one that is already there."""
    if html_path is None:
        return None
    from provibench.bench.html_report import render_html

    target = _output_target(invocation, html_path, force, option="html")
    target.write_text(render_html(document), encoding="utf-8")
    return str(target)


def _write_markdown(
    invocation: Invocation, md_path: str | None, text: str, force: bool
) -> str | None:
    if md_path is None:
        return None
    target = _output_target(invocation, md_path, force, option="md")
    target.write_text(text + "\n", encoding="utf-8")
    return str(target)


def _output_target(invocation: Invocation, path: str, force: bool, *, option: str) -> Path:
    """Resolve a written report's path, or refuse to replace one that is already there.

    Both written forms take the same `--force`: a report is regenerated from the same run,
    so silently replacing either file is the same surprise.
    """
    target = Path(path)
    if not target.is_absolute():
        target = invocation.cwd / target
    if target.exists() and not force:
        raise PreconditionFailed(
            f"{target} already exists",
            hint=f"Pass --force to overwrite the --{option} output",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_run_dir(run: str, runs_dir: Path, cwd: Path) -> Path:
    candidate = Path(run)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if (candidate / "run.json").is_file():
        return candidate
    if run == "latest":
        newest = _newest_run_dir(runs_dir.glob("*/*"))
    else:
        trace_dir = runs_dir / run
        newest = _newest_run_dir(trace_dir.glob("*")) if trace_dir.is_dir() else None
    if newest is None:
        raise NotFound(
            f"No run found for {run!r}",
            hint=f"Pass a run directory, a trace name under {runs_dir}, or 'latest'",
        )
    return newest


def _newest_run_dir(candidates: Iterable[Path]) -> Path | None:
    runs = sorted(
        (path for path in candidates if path.is_dir() and (path / "run.json").is_file()),
        key=lambda path: path.name,
    )
    return runs[-1] if runs else None
