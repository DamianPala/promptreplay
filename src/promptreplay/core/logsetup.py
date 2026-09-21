"""Diagnostics on stderr through the standard `logging` module (O3)."""

import logging
from typing import TextIO

from promptreplay.core.terminal_text import escape_terminal_text


class TerminalTextFormatter(logging.Formatter):
    """Escape untrusted control characters after logging formats a diagnostic."""

    def format(self, record: logging.LogRecord) -> str:
        return escape_terminal_text(super().format(record))


def configure_logging(name: str, stderr: TextIO, *, quiet: bool, verbose: bool) -> None:
    """Route the tool's logger to stderr at the level `--quiet` and `--verbose` select.

    `--quiet` wins over `--verbose` and silences every record; the F3 object is written
    outside the logging module so it survives.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stderr)
    handler.setFormatter(TerminalTextFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    if quiet:
        logger.setLevel(logging.CRITICAL + 1)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
