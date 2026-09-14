"""`scrub`: write a shareable copy of a trace and report what it changed.

`provibench.bench.*` pulls in pydantic; its symbols are imported only inside the callback,
so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

# Only stdlib-level names: `bench.scrub` imports no pydantic, so the schema can name the
# three values `select_entries` returns without loading anything heavy at parser build.
from provibench.bench.scrub import SELECTION_ALL, SELECTION_CONVERSATION, SELECTION_ORDER
from provibench.commands.inspect import (
    load_trace_documents,
    load_trace_text,
    resolve_trace_path,
)
from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    array,
    as_document,
    as_list,
    boolean,
    integer,
    obj,
    string,
)
from provibench.core.errors import InvalidInput, PreconditionFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.scrub import Replacement, ScrubReport

_RULE = obj({"rule": string(), "count": integer()}, required=["rule", "count"])
_OUTPUT = obj(
    {
        "trace": string(),
        "out": string(),
        "entries_in": integer(),
        "entries_out": integer(),
        "entries_dropped": integer(),
        "selection": string(enum=[SELECTION_ALL, SELECTION_CONVERSATION, SELECTION_ORDER]),
        "bytes_in": integer(),
        "bytes_out": integer(),
        "user_names": array(string()),
        "rules": array(_RULE),
        "changed": boolean(),
    },
    required=[
        "trace",
        "out",
        "entries_in",
        "entries_out",
        "entries_dropped",
        "selection",
        "bytes_in",
        "bytes_out",
        "user_names",
        "rules",
        "changed",
    ],
)

_SELECTIONS = {
    SELECTION_ALL: "every entry",
    SELECTION_CONVERSATION: "the first N of the main conversation",
    SELECTION_ORDER: "the first N in file order (this trace has no conversation keys)",
}


def render_scrub(invocation: Invocation, document: Document) -> None:
    """The report as a rules table plus one summary block."""
    from rich import box
    from rich.markup import escape
    from rich.table import Table

    def cell(value: object) -> str:
        return escape(escape_terminal_text("-" if value is None else str(value)))

    console = invocation.stdout_console()
    table = Table(box=box.SIMPLE, header_style="bold", title="Scrubbed")
    table.add_column("rule")
    table.add_column("count", justify="right")
    for raw in as_list(document.get("rules")) or []:
        row = as_document(raw)
        if row is not None:
            table.add_row(cell(row.get("rule")), cell(row.get("count")))
    console.print(table)

    names = ", ".join(str(name) for name in as_list(document.get("user_names")) or [])
    console.print(f"User names: {cell(names) if names else '(none)'}")
    selection = _SELECTIONS.get(str(document.get("selection")), "-")
    console.print(
        f"Entries: {cell(document.get('entries_in'))} -> {cell(document.get('entries_out'))} "
        f"({cell(document.get('entries_dropped'))} dropped)"
    )
    console.print(f"Selection: {escape(selection)}")
    console.print(f"Bytes: {cell(document.get('bytes_in'))} -> {cell(document.get('bytes_out'))}")
    console.print(f"{cell(document.get('trace'))} -> {cell(document.get('out'))}")


@click.command(
    "scrub",
    cls=Command,
    spec=CommandSpec(effects=Effects.IDEMPOTENT, output=_OUTPUT, render=render_scrub),
    help="Write a shareable copy of a trace, with home paths, secrets and metadata removed.\n\n"
    "TRACE is either an existing path, a name under traces_dir, or 'sample' (the packaged "
    "example). The copy goes to --out, gzip-compressed when that path ends in .gz, and the "
    "report lists what each rule replaced. Project-specific names cannot be recognised: "
    "remove those with --replace OLD=NEW, and operator names with --user NAME.",
)
@click.argument("trace", help="Trace path, a name under traces_dir, or 'sample'")
@click.option(
    "--out",
    "out_path",
    required=True,
    metavar="PATH",
    help="Where to write the scrubbed trace; a .gz path is compressed",
)
@click.option(
    "--replace",
    "replace_specs",
    multiple=True,
    metavar="OLD=NEW",
    help="Literal replacement applied to every string after the built-in rules; repeatable",
)
@click.option(
    "--user",
    "users",
    multiple=True,
    metavar="NAME",
    help="Replace whole-word NAME with 'user'; repeatable, for `ls -l` owner columns and prose",
)
@click.option(
    "--turns",
    type=click.IntRange(1),
    default=None,
    help="Keep only the first N entries of the main conversation, dropping the rest",
)
@click.option(
    "--allow-email",
    "allow_emails",
    multiple=True,
    metavar="ADDR",
    help="E-mail address to keep; repeatable",
)
@click.option("--force", is_flag=True, help="Overwrite an existing --out")
@click.pass_context
def scrub(
    ctx: click.Context,
    *,
    trace: str,
    out_path: str,
    replace_specs: tuple[str, ...],
    users: tuple[str, ...],
    turns: int | None,
    allow_emails: tuple[str, ...],
    force: bool,
) -> Document:
    from provibench.bench.scrub import (
        Scrubber,
        ScrubError,
        collect_user_names,
        scrub_entry,
        select_entries,
    )
    from provibench.bench.trace import dump_document, write_trace_text

    invocation = require_invocation(ctx)
    trace_path = resolve_trace_path(trace, invocation)
    destination = _resolve_out(out_path, invocation)
    if destination.resolve() == trace_path.resolve():
        raise InvalidInput(
            "--out must differ from TRACE", hint="Pass another output path for the copy"
        )
    if destination.exists() and not force:
        raise PreconditionFailed(
            f"{destination} already exists", hint="Pass --force to overwrite it"
        )
    if any(not name for name in users):
        raise InvalidInput("--user needs a non-empty NAME")

    replacements = tuple(_parse_replacement(spec) for spec in replace_specs)
    text = load_trace_text(trace_path)
    documents = load_trace_documents(trace_path)
    kept, selection = select_entries(documents, turns=turns)

    scrubber = Scrubber(
        allow_emails=allow_emails,
        replacements=replacements,
        users=users,
        home_names=collect_user_names(kept),
    )
    try:
        scrubbed = [scrub_entry(document, scrubber) for document in kept]
    except ScrubError as exc:
        raise InvalidInput(str(exc)) from exc
    payload = "".join(f"{dump_document(document)}\n" for document in scrubbed)

    destination.parent.mkdir(parents=True, exist_ok=True)
    write_trace_text(destination, payload)
    report = scrubber.report(
        entries_in=len(documents),
        entries_out=len(scrubbed),
        bytes_in=len(text.encode("utf-8")),
        bytes_out=len(payload.encode("utf-8")),
    )
    return _document(trace_path, destination, report, selection)


def _document(trace_path: Path, destination: Path, report: ScrubReport, selection: str) -> Document:
    return {
        "trace": str(trace_path),
        "out": str(destination),
        "entries_in": report.entries_in,
        "entries_out": report.entries_out,
        "entries_dropped": report.entries_dropped,
        "selection": selection,
        "bytes_in": report.bytes_in,
        "bytes_out": report.bytes_out,
        "user_names": list(report.user_names),
        "rules": [{"rule": row.rule, "count": row.count} for row in report.rules],
        "changed": report.changed,
    }


def _resolve_out(out_path: str, invocation: Invocation) -> Path:
    destination = Path(out_path)
    if not destination.is_absolute():
        destination = invocation.cwd / destination
    return destination


def _parse_replacement(spec: str) -> Replacement:
    """`OLD=NEW` as a replacement; OLD may not be empty, NEW may be (a deletion)."""
    from provibench.bench.scrub import Replacement

    old, separator, new = spec.partition("=")
    if not separator or not old:
        raise InvalidInput(
            f"--replace needs OLD=NEW, got {spec!r}",
            hint="For example --replace acme-corp=example",
        )
    return Replacement(old=old, new=new)
