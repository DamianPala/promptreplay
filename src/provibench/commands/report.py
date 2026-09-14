"""`report`: summarise an existing run, probe or full replay, optionally writing markdown.

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
from provibench.core.errors import NotFound
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
        "protocol": string(enum=["full", "probe"]),
        "options": _OPTIONS,
        "summaries": array(SUMMARY),
        "probe_summaries": array(PROBE_SUMMARY),
        "markdown_path": nullable_string(),
        "changed": boolean(),
    },
    required=[
        "run_dir",
        "trace",
        "conversation",
        "created",
        "protocol",
        "options",
        "summaries",
        "probe_summaries",
        "markdown_path",
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
    "With --md, the same report is also written as markdown.",
)
@click.argument("run")
@click.option("--md", "md_path", default=None, help="Write the report as markdown to this path")
@click.pass_context
def report(ctx: click.Context, run: str, md_path: str | None) -> Document:
    invocation = require_invocation(ctx)
    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = _resolve_run_dir(run, runs_dir, invocation.cwd)

    from provibench.bench.probe_runs import read_protocol

    if read_protocol(run_dir) == "probe":
        return _probe_report(invocation, run_dir, md_path)
    return _full_report(invocation, run_dir, md_path)


def _full_report(invocation: Invocation, run_dir: Path, md_path: str | None) -> Document:
    from provibench.bench.replay import load_run
    from provibench.bench.summary import cache_mode_note, render_markdown, summarize

    meta, results = load_run(run_dir)
    notes = [cache_mode_note(meta.options.warm)]
    summaries = [
        summarize(ref.label, results[ref.label], ref.providers, notes) for ref in meta.runs
    ]
    markdown_path = _write_markdown(invocation, md_path, render_markdown(summaries))
    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "protocol": meta.protocol,
        "options": _replay_options(meta.options),
        "summaries": [summary_to_document(s) for s in summaries],
        "probe_summaries": [],
        "markdown_path": markdown_path,
        "changed": markdown_path is not None,
    }


def _probe_report(invocation: Invocation, run_dir: Path, md_path: str | None) -> Document:
    from provibench.bench.probe_runs import load_probe_run
    from provibench.bench.probe_summary import probe_markdown, summarize_probe
    from provibench.bench.summary import cache_mode_note

    meta, records = load_probe_run(run_dir)
    notes = [*meta.notes, cache_mode_note(meta.options.warm)]
    summaries = [
        summarize_probe(
            ref.label, records[ref.label], prices=meta.prices.get(ref.label), notes=notes
        )
        for ref in meta.specs
    ]
    markdown_path = _write_markdown(invocation, md_path, probe_markdown(summaries))
    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "protocol": meta.protocol,
        "options": dict(meta.options.model_dump()),
        "summaries": [],
        "probe_summaries": [probe_summary_to_document(s) for s in summaries],
        "markdown_path": markdown_path,
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


def _write_markdown(invocation: Invocation, md_path: str | None, text: str) -> str | None:
    if md_path is None:
        return None
    target = Path(md_path)
    if not target.is_absolute():
        target = invocation.cwd / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    return str(target)


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
