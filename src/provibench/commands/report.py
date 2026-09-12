"""`report`: summarise an existing replay run, optionally writing it out as markdown.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside
the callback, so building the CLI (schema, --help, completion) stays cheap.
"""

from collections.abc import Iterable
from pathlib import Path

import click

from provibench.commands.summary_view import SUMMARY, render_summaries, summary_to_document
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

_OPTIONS = obj(
    {
        "max_tokens": integer(),
        "delay_s": number(),
        "strip_thinking": boolean(),
        "timeout_s": number(),
    },
    required=["max_tokens", "delay_s", "strip_thinking", "timeout_s"],
)
_OUTPUT = obj(
    {
        "run_dir": string(),
        "trace": string(),
        "conversation": string(),
        "created": string(),
        "options": _OPTIONS,
        "summaries": array(SUMMARY),
        "markdown_path": nullable_string(),
        "changed": boolean(),
    },
    required=[
        "run_dir",
        "trace",
        "conversation",
        "created",
        "options",
        "summaries",
        "markdown_path",
        "changed",
    ],
)


@click.command(
    "report",
    cls=Command,
    spec=CommandSpec(effects=Effects.IDEMPOTENT, output=_OUTPUT, render=render_summaries),
    help="Summarise an existing replay run.\n\n"
    "RUN is a run directory, a trace name (the newest run under <runs-dir>/<name>), or "
    "'latest' (the newest run across every trace). With --md, the same report is also "
    "written as markdown.",
)
@click.argument("run")
@click.option("--md", "md_path", default=None, help="Write the report as markdown to this path")
@click.pass_context
def report(ctx: click.Context, run: str, md_path: str | None) -> Document:
    from provibench.bench.replay import load_run
    from provibench.bench.summary import render_markdown, summarize

    invocation = require_invocation(ctx)
    runs_dir = Path(invocation.setting("runs_dir") or ".")
    run_dir = _resolve_run_dir(run, runs_dir, invocation.cwd)
    meta, results = load_run(run_dir)
    summaries = [summarize(ref.label, results[ref.label], ref.providers) for ref in meta.runs]

    markdown_path: str | None = None
    if md_path:
        target = Path(md_path)
        if not target.is_absolute():
            target = invocation.cwd / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_markdown(summaries) + "\n", encoding="utf-8")
        markdown_path = str(target)

    return {
        "run_dir": str(run_dir),
        "trace": meta.trace,
        "conversation": meta.conversation,
        "created": meta.created,
        "options": {
            "max_tokens": meta.options.max_tokens,
            "delay_s": meta.options.delay_s,
            "strip_thinking": meta.options.strip_thinking,
            "timeout_s": meta.options.timeout_s,
        },
        "summaries": [summary_to_document(s) for s in summaries],
        "markdown_path": markdown_path,
        "changed": markdown_path is not None,
    }


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
