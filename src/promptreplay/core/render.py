"""Turning a command's result into bytes on stdout in the selected format.

JSON and NDJSON go through `promptreplay.core.output` without rich. Human-readable output uses
rich, imported inside the functions that need it, so `--json` never loads it.
"""

import shlex
from collections.abc import Iterator
from typing import cast

from promptreplay.core.context import Invocation
from promptreplay.core.documents import Document, as_document, as_list, sanitize_document
from promptreplay.core.output import Format, write_document, write_plain_line
from promptreplay.core.spec import CommandSpec, Result
from promptreplay.core.terminal_text import escape_terminal_text


def emit_result(invocation: Invocation, spec: CommandSpec, result: Result) -> None:
    """Write `result` to stdout in the invocation's selected format."""
    if result is None:
        return
    if invocation.output_file is not None:
        target = invocation.output_file
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output, invocation.rendering_to(output):
            _emit_result(invocation, spec, result)
        return
    _emit_result(invocation, spec, result)


def _emit_result(invocation: Invocation, spec: CommandSpec, result: Result) -> None:
    """Emit a result to the invocation's current stdout stream."""
    if isinstance(result, dict):
        _emit_document(invocation, spec, result)
        return
    records = cast(Iterator[Document], result)
    stdout = invocation.streams.stdout
    match invocation.format:
        case Format.NDJSON | Format.JSON:
            for record in records:
                write_document(stdout, record)  # sanitizes internally
        case Format.PLAIN:
            for record in records:
                write_plain_line(stdout, sanitize_document(record)[spec.plain_field or ""])
        case _:
            _render_records(invocation, spec, (sanitize_document(record) for record in records))


def _emit_document(invocation: Invocation, spec: CommandSpec, document: Document) -> None:
    stdout = invocation.streams.stdout
    match invocation.format:
        case Format.JSON:
            indent = 2 if stdout.isatty() else None
            write_document(stdout, document, indent=indent)  # sanitizes internally
        case Format.NDJSON:
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
    if "items" in document:
        _render_table(invocation, _items(document))
        if document.get("has_more"):
            cursor = escape_terminal_text(str(document.get("next_cursor")))
            stdout = invocation.streams.stdout
            stdout.write(f"More items available. Continue with --cursor {cursor}\n")
            stdout.flush()
        return
    console = invocation.stdout_console()
    for key, value in document.items():
        parts = as_list(value)
        if key == "next" and parts is not None:
            console.print(f"Next: {_escape(shlex.join(str(part) for part in parts))}")
        else:
            console.print(f"[bold]{_escape(_label(key))}:[/bold] {_escape(format_value(value))}")


def _items(document: Document) -> list[Document]:
    entries = as_list(document.get("items")) or []
    return [item for item in map(as_document, entries) if item is not None]


def _render_table(invocation: Invocation, rows: list[Document]) -> None:
    """A page of items as a plain fixed-width table, or `No items` when there are none."""
    from promptreplay.core.text_table import text_table

    stdout = invocation.streams.stdout
    if not rows:
        stdout.write("No items\n")
        stdout.flush()
        return
    columns = list(rows[0].keys())
    lines = text_table(
        [_label(column) for column in columns],
        [[format_value(row.get(column)) for column in columns] for row in rows],
    )
    stdout.write("\n".join(escape_terminal_text(line) for line in lines) + "\n")
    stdout.flush()


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
