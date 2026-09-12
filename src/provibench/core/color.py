"""Color policy (H4): optional, per stream, overridable only on human-readable output."""

from collections.abc import Mapping
from enum import StrEnum


class ColorMode(StrEnum):
    """Values of `--color`."""

    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


def color_enabled(mode: ColorMode, *, isatty: bool, env: Mapping[str, str]) -> bool:
    """Whether a human-readable stream may carry color.

    `auto` disables color when `NO_COLOR` is non-empty, `TERM` is `dumb`, or the stream
    is not a TTY; `always` and `never` override that default.
    """
    match mode:
        case ColorMode.ALWAYS:
            return True
        case ColorMode.NEVER:
            return False
        case ColorMode.AUTO:
            return isatty and not env.get("NO_COLOR") and env.get("TERM") != "dumb"
