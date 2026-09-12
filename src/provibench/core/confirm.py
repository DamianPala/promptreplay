"""The confirmation gate for wide or irreversible mutations (R3)."""

from provibench.core.context import Invocation
from provibench.core.errors import ConfirmationRequired
from provibench.core.terminal_text import escape_terminal_text


def require_confirmation(invocation: Invocation, *, question: str, yes: bool) -> None:
    """Return when the caller consented with `--yes` or at a prompt; otherwise fail closed.

    The prompt is written to stderr only in an interactive context with a TTY stderr and
    a free stdin; anything else, an unanswered prompt included, is `confirmation_required`.
    """
    if yes:
        return
    if not invocation.can_prompt():
        raise ConfirmationRequired(f"{question} requires confirmation", hint="Repeat with --yes")
    stderr = invocation.streams.stderr
    stderr.write(f"{escape_terminal_text(question)}? [y/N] ")
    stderr.flush()
    answer = invocation.streams.stdin.readline().strip().lower()
    if answer not in ("y", "yes"):
        raise ConfirmationRequired("Confirmation declined", hint="Repeat with --yes to skip it")
