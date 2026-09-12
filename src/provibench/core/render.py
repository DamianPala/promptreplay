"""Turning a command's result into bytes on stdout in the selected format.

JSON and NDJSON go through `provibench.core.output` without rich. Human-readable output uses
rich, imported inside the functions that need it, so `--json` never loads it.
"""

import shlex
from collections.abc import Iterator
from typing import TYPE_CHECKING

from provibench.core.context import Invocation
from provibench.core.documents import Document, as_document, as_list, sanitize_document
from provibench.core.output import Format, write_document, write_plain_line
from provibench.core.spec import CommandSpec, Result
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from rich.console import Console


def emit_result(invocation: Invocation, spec: CommandSpec, result: Result) -> None:
    """Write `result` to stdout in the invocation's selected format."""
    if result is None:
        return
    if isinstance(result, dict):
        _emit_document(invocation, spec, result)
        return
    stdout = invocation.streams.stdout
    match invocation.format:
        case Format.NDJSON | Format.JSON:
            for record in result:
                write_document(stdout, record)  # sanitizes internally
        case Format.PLAIN:
            for record in result:
                write_plain_line(stdout, sanitize_document(record)[spec.plain_field or ""])
        case _:
            _render_records(invocation, spec, (sanitize_document(record) for record in result))


def _emit_document(invocation: Invocation, spec: CommandSpec, document: Document) -> None:
    stdout = invocation.streams.stdout
    match invocation.format:
        case Format.JSON | Format.NDJSON:
            write_document(stdout, document)  # sanitizes internally
        case Format.PLAIN:
            document = sanitize_document(document)
            for item in _items(document):
                write_plain_line(stdout, item[spec.plain_field or ""])
        case _:
            document = sanitize_document(document)
            if spec.render is not None:
                spec.render(invocation, document)
            else:
                render_document(invocation, document)


def render_document(invocation: Invocation, document: Document) -> None:
    """The generic human-readable rendering: a table for pages, `Key: value` lines otherwise."""
    console = invocation.stdout_console()
    if "items" in document:
        _render_table(console, _items(document))
        if document.get("has_more"):
            cursor = document.get("next_cursor")
            console.print(f"More items available. Continue with --cursor {_escape(str(cursor))}")
        return
    for key, value in document.items():
        parts = as_list(value)
        if key == "next" and parts is not None:
            console.print(f"Next: {_escape(shlex.join(str(part) for part in parts))}")
        else:
            console.print(f"[bold]{_escape(_label(key))}:[/bold] {_escape(format_value(value))}")


def _items(document: Document) -> list[Document]:
    entries = as_list(document.get("items")) or []
    return [item for item in map(as_document, entries) if item is not None]


def _render_table(console: "Console", rows: list[Document]) -> None:
    from rich import box
    from rich.table import Table

    if not rows:
        console.print("No items")
        return
    table = Table(box=box.SIMPLE, header_style="bold")
    columns = list(rows[0].keys())
    for column in columns:
        table.add_column(_escape(_label(column)))
    for row in rows:
        table.add_row(*(_escape(format_value(row.get(column))) for column in columns))
    console.print(table)


def _render_records(invocation: Invocation, spec: CommandSpec, records: Iterator[Document]) -> None:
    if spec.render is None:
        raise RuntimeError("Human-readable stream rendering is not configured")
    for record in records:
        spec.render(invocation, record)


def format_value(value: object) -> str:
    """A person-facing rendering of one JSON value."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    parts = as_list(value)
    if parts is not None:
        return ", ".join(format_value(part) for part in parts) or "(none)"
    return str(value)


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _escape(text: str) -> str:
    from rich.markup import escape

    return escape(escape_terminal_text(text))
