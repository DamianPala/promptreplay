"""Parameter types that carry what the standard needs beyond click's defaults.

`StdinOrPath` marks a flag whose `-` selects stdin (I1 `accepts_stdin`). `DurationType` is
the I8b duration syntax and `limit_type` the bounded `--limit` of I7a and O6b. Positional
arguments use `click.Argument` directly: since click 8.5 it takes `help`, which D8 reads.
"""

import re
from math import isfinite
from typing import override

import click

from provibench.core.errors import InvalidInput

DURATION_PATTERN = re.compile(r"^([1-9][0-9]*)([smh])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}

LIMIT_MIN = 1
LIMIT_MAX = 100
STDIN_MARKER = "-"


class StdinOrPath(click.ParamType[str]):
    """A file path where `-` selects stdin; introspection reports `accepts_stdin: true`."""

    name = "path"

    @override
    def convert(
        self, value: object, param: click.Parameter | None, ctx: click.Context | None
    ) -> str:
        return str(value)


STDIN_OR_PATH = StdinOrPath()


def parse_duration(text: str) -> int:
    """Seconds for a duration such as `30s`, `5m`, or `2h`; anything else is `InvalidInput`."""
    match = DURATION_PATTERN.match(text)
    if match is None:
        raise InvalidInput(
            f"Invalid duration {text!r}: use a positive integer followed by s, m, or h",
        )
    try:
        seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2)]
        if not isfinite(float(seconds)):
            raise OverflowError
    except (ValueError, OverflowError) as error:
        raise InvalidInput(
            "Duration must fit a finite number of seconds; pass a smaller duration"
        ) from error
    return seconds


class DurationType(click.ParamType[int]):
    """The I8b duration syntax as a click parameter type; converts to seconds."""

    name = "duration"

    @override
    def convert(
        self, value: object, param: click.Parameter | None, ctx: click.Context | None
    ) -> int:
        if isinstance(value, int):
            return value
        try:
            return parse_duration(str(value))
        except InvalidInput as error:
            self.fail(error.message, param, ctx)


DURATION = DurationType()


def limit_type() -> click.IntRange:
    """The accepted `--limit` range; values outside it are usage errors."""
    return click.IntRange(LIMIT_MIN, LIMIT_MAX)
