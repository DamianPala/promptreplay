"""Output format selection (O2) and the machine-readable writers (O5, O7).

The JSON and NDJSON paths write `json.dumps` output straight to stdout; rich is never
involved, so `--json` pays nothing for it.
"""

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TextIO

from provibench.core.documents import Document, sanitize_document
from provibench.core.errors import InvalidInput
from provibench.core.terminal_text import escape_terminal_text


class Format(StrEnum):
    """Output formats the tool can select for stdout."""

    TEXT = "text"
    JSON = "json"
    NDJSON = "ndjson"
    PLAIN = "plain"
    MARKDOWN = "md"
    HTML = "html"

    @property
    def machine_readable(self) -> bool:
        """`json` and `ndjson` are machine-readable; everything else is for people."""
        return self in (Format.JSON, Format.NDJSON)


@dataclass(frozen=True, slots=True)
class FormatDefaults:
    """The O2 default format for each stdout context."""

    tty: Format
    non_tty: Format

    def to_document(self) -> Document:
        """The D7 or D8 `format_defaults` object."""
        return {"tty": self.tty.value, "non_tty": self.non_tty.value}


TOOL_FORMAT_DEFAULTS = FormatDefaults(tty=Format.TEXT, non_tty=Format.JSON)
STREAM_FORMAT_DEFAULTS = FormatDefaults(tty=Format.TEXT, non_tty=Format.NDJSON)


def select_format(
    defaults: FormatDefaults,
    *,
    json_flag: bool,
    plain_flag: bool,
    named_format: str | None = None,
    stream: bool,
    stdout_isatty: bool,
) -> Format:
    """The stdout format for one call: explicit flags first, then the context default."""
    chosen = [
        flag
        for flag, set_ in (
            ("--json", json_flag),
            ("--plain", plain_flag),
            (f"--format {named_format}", named_format is not None),
        )
        if set_
    ]
    if len(chosen) > 1:
        raise InvalidInput(
            f"{' and '.join(chosen)} select different output formats",
            hint="Pass one of them",
        )
    if plain_flag:
        return Format.PLAIN
    if json_flag:
        return Format.NDJSON if stream else Format.JSON
    if named_format is not None:
        try:
            return Format(named_format)
        except ValueError as exc:
            known = ", ".join(member.value for member in Format)
            raise InvalidInput(
                f"--format {named_format!r} is not an output format", hint=f"One of: {known}"
            ) from exc
    return defaults.tty if stdout_isatty else defaults.non_tty


def write_document(stdout: TextIO, document: Document) -> None:
    """One JSON value, UTF-8, on one LF-terminated line (O5a).

    `sanitize_document` replaces any lone surrogate (from a path or other OS-supplied
    string decoded with `surrogateescape`) with U+FFFD first, so the written text is
    always valid UTF-8 instead of the raw invalid byte a surrogate-escaped stream would
    otherwise re-emit. Flushed at once, so a record of an NDJSON stream is readable
    before the next wait (O7a).
    """
    stdout.write(json.dumps(sanitize_document(document), ensure_ascii=False))
    stdout.write("\n")
    stdout.flush()


def write_plain_line(stdout: TextIO, value: object) -> None:
    """One H5 item per LF-terminated line, without decoration."""
    stdout.write(f"{escape_terminal_text(str(value))}\n")


def silence_stdout(stdout: TextIO) -> None:
    """Point stdout at the null device after a broken pipe so the interpreter's final flush
    does not print a second broken-pipe diagnostic (O8)."""
    try:
        descriptor = stdout.fileno()
    except (OSError, ValueError):
        # An in-memory stream has no descriptor and nothing to flush at exit.
        return
    os.dup2(os.open(os.devnull, os.O_WRONLY), descriptor)
