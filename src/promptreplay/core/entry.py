"""The tool-wide entry point: exit codes (F1) and the single error location (F2).

`Root.run` is the only place that turns an exception into an exit status. It drives click
through `make_context` and `invoke` rather than `main`, because `main` writes a blank line
to the real `sys.stderr` on an interrupt and exits 1 on a closed pipe; owning the dispatch
keeps every stream injected and every exit code in one ladder. Usage errors from click
become `invalid_input` with exit 2; every `CliError` maps to its own exit code and, when
the context demands it, to an F3 object on the last non-empty stderr line.
"""

from collections.abc import Sequence
from typing import override

import click

from promptreplay.core.completion import completion_variable
from promptreplay.core.context import Invocation
from promptreplay.core.errors import CliError, Interrupted, InvalidInput
from promptreplay.core.output import silence_stdout, write_document
from promptreplay.core.registry import Group


class Root(Group):
    """The root command group of a tool."""

    def run(self, argv: Sequence[str], invocation: Invocation) -> int:
        """Parse and run `argv`; return the process exit code without raising."""
        invocation.argv = list(argv)
        try:
            return self._dispatch(argv, invocation)
        except click.exceptions.Exit as exit_request:
            # `--help` and `--version` end the call this way after writing their output.
            return exit_request.exit_code
        except click.UsageError as error:
            return report_error(invocation, _usage_error(invocation, error))
        except click.ClickException as error:
            return report_error(invocation, InvalidInput(error.format_message()))
        except (click.Abort, KeyboardInterrupt, EOFError):
            return report_error(invocation, Interrupted("The command was interrupted"))
        except CliError as error:
            return report_error(invocation, error)
        except BrokenPipeError:
            # A downstream reader closed stdout; that is not a failure of this tool (O8).
            silence_stdout(invocation.streams.stdout)
            return 0

    def _dispatch(self, argv: Sequence[str], invocation: Invocation) -> int:
        """Answer a shell completion request, or parse and invoke the command."""
        variable = completion_variable(invocation.program)
        instruction = invocation.env.get(variable)
        if instruction:
            from click.shell_completion import shell_complete

            return shell_complete(self, {}, invocation.program, variable, instruction)
        with self.make_context(invocation.program, list(argv), obj=invocation) as ctx:
            self.invoke(ctx)
        return 0

    @override
    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if self.epilog:
            formatter.write_paragraph()
            formatter.write_text(self.epilog)


def report_error(invocation: Invocation, error: CliError) -> int:
    """Write the error to stderr as F2 requires and return its exit code.

    The F3 object is written whenever the call is machine-readable or stderr is not a
    terminal. The `Error:` and `Hint:` lines precede it only when someone can read them:
    a machine-readable call with a captured stderr gets the object alone.
    """
    stderr = invocation.streams.stderr
    machine = invocation.machine_readable
    if not machine or stderr.isatty():
        invocation.message(f"Error: {error.message}", style="bold red")
        if error.hint is not None:
            invocation.message(f"Hint: {error.hint}")
    if machine or not stderr.isatty() or invocation.flags.quiet:
        write_document(stderr, error.to_document())
    return error.exit_code


def _usage_error(invocation: Invocation, error: click.UsageError) -> InvalidInput:
    """click's usage error as an F3 `invalid_input` with a hint at the right help page."""
    path = error.ctx.command_path if error.ctx is not None else invocation.program
    hint = f"Run {path} --help"
    if isinstance(error, click.exceptions.NoArgsIsHelpError):
        return InvalidInput("Missing command", hint=hint)
    return InvalidInput(error.format_message(), hint=hint)
