"""Bounded UTF-8 text input mechanics for caller-owned input policies."""

from pathlib import Path
from typing import TextIO


class InputTooLarge(Exception):
    """The source exceeds its caller-provided byte limit."""


class InvalidUtf8(Exception):
    """The source cannot be represented as strict UTF-8 text."""


def read_bounded_file(path: Path, *, max_bytes: int) -> str:
    """Read at most ``max_bytes`` of strict UTF-8 text from ``path``."""
    with path.open("rb") as stream:
        return _decode(stream.read(max_bytes + 1), max_bytes=max_bytes)


def read_bounded_stream(stream: TextIO, *, max_bytes: int) -> str:
    """Read bounded strict UTF-8 text from an injected text stream."""
    try:
        data = stream.read(max_bytes + 1).encode("utf-8", "surrogateescape")
    except UnicodeError as error:
        raise InvalidUtf8 from error
    return _decode(data, max_bytes=max_bytes)


def _decode(data: bytes, *, max_bytes: int) -> str:
    if len(data) > max_bytes:
        raise InputTooLarge
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidUtf8 from error
