"""The command registry: click objects that carry the standard's metadata.

`Command` attaches a `CommandSpec` to the object the parser dispatches and runs the
format selection, `--plain` gate, I3b stdin gate, and result emission around the callback.
`Group` and `Command` both accept the D7 global flags, so they work before or after the
command path.
"""

from collections.abc import Callable, MutableMapping
from typing import override

import click

from provibench.core.color import ColorMode
from provibench.core.context import Invocation
from provibench.core.errors import InvalidInput
from provibench.core.output import select_format
from provibench.core.params import StdinOrPath
from provibench.core.render import emit_result
from provibench.core.spec import CommandSpec

PLAIN_META = f"{__name__}:plain"
"""Context meta key under which `--plain` records its value (see `provibench.core.paging`).

Derived from the module name, so a renamed package keeps the key private without an edit.
"""

PAGE_PARAMETERS = ("limit", "cursor")
"""Parameters whose explicit presence makes `--plain` acceptable (H5a)."""

GLOBAL_FLAG_NAMES = ("json", "quiet", "verbose", "color", "config", "targets")
"""Parser names of the D7 global flags; command-specific flags must not reuse them."""


def _record_global_flag(ctx: click.Context, param: click.Parameter, value: object) -> object:
    """Store a global flag given on the command line, at any level of the path."""
    if ctx.get_parameter_source(param.name or "") is not click.core.ParameterSource.COMMANDLINE:
        return value
    invocation = ctx.find_object(Invocation)
    if invocation is not None:
        invocation.flags.record(param.name or "", value)
    return value


def global_options() -> list[click.Option]:
    """Fresh instances of the D7 global flags."""
    return [
        click.Option(
            ["--json"],
            is_flag=True,
            expose_value=False,
            callback=_record_global_flag,
            help="Emit JSON (NDJSON for a record stream)",
        ),
        click.Option(
            ["--quiet", "-q"],
            is_flag=True,
            expose_value=False,
            callback=_record_global_flag,
            help="Suppress stderr diagnostics except the error object; wins over --verbose",
        ),
        click.Option(
            ["--verbose", "-v"],
            is_flag=True,
            expose_value=False,
            callback=_record_global_flag,
            help="Log DEBUG diagnostics on stderr; long waits report after about 10 seconds "
            "and at least every 30 seconds",
        ),
        click.Option(
            ["--color"],
            type=click.Choice([mode.value for mode in ColorMode]),
            default=ColorMode.AUTO.value,
            metavar="WHEN",
            expose_value=False,
            callback=_record_global_flag,
            help="Color human-readable output: auto (default) is per stream and disables color "
            "without a TTY, with TERM=dumb, or a nonempty NO_COLOR; `always` forces color; "
            "`never` turns color off",
        ),
        click.Option(
            ["--config", "-c"],
            metavar="PATH",
            expose_value=False,
            callback=_record_global_flag,
            help="Configuration file at most 65536 UTF-8 bytes. A flag or environment selection "
            "replaces the default user file; its path resolves from the working directory. When "
            "omitted, a default user file may apply; config show reports effective paths and "
            "sources",
        ),
        click.Option(
            ["--targets"],
            metavar="PATH",
            expose_value=False,
            callback=_record_global_flag,
            help="Targets file listing configured providers, models, and API key sources. A flag "
            "or environment selection replaces the packaged default; its path resolves from the "
            "working directory. When omitted, a selected configuration or the packaged default "
            "applies; config show reports effective paths and sources",
        ),
    ]


def is_global_flag(param: click.Parameter) -> bool:
    """Whether `param` is one of the D7 global flags."""
    return isinstance(param, click.Option) and param.name in GLOBAL_FLAG_NAMES


def format_parameters(
    command: click.Command, ctx: click.Context, formatter: click.HelpFormatter
) -> None:
    """Help sections for the command's own options and for the global flags (D3b).

    click 8.5 prints positional arguments with their help in its own `Positional arguments`
    section, but it lists the global flags among the command's own options; two sections
    say which flags belong to the command and which to the tool.
    """
    own: list[tuple[str, str]] = []
    shared: list[tuple[str, str]] = []
    for parameter in command.get_params(ctx):
        record = parameter.get_help_record(ctx)
        if record is None or isinstance(parameter, click.Argument):
            continue
        (shared if is_global_flag(parameter) else own).append(record)
    if own:
        with formatter.section("Options"):
            formatter.write_dl(own)
    if shared:
        with formatter.section("Global options (before or after the command path)"):
            formatter.write_dl(shared)


def require_invocation(ctx: click.Context) -> Invocation:
    """The invocation on the context; absent only when the entry point was bypassed."""
    invocation = ctx.find_object(Invocation)
    if invocation is None:
        raise RuntimeError("The click context carries no Invocation; use Root.run")
    return invocation


class Command(click.Command):
    """A command with a `CommandSpec`; selects the format, runs the callback, emits the result."""

    def __init__(
        self,
        name: str | None,
        *,
        spec: CommandSpec,
        callback: Callable[..., object] | None = None,
        params: list[click.Parameter] | None = None,
        help: str | None = None,
        hidden: bool = False,
    ) -> None:
        super().__init__(name, callback=callback, params=params, help=help, hidden=hidden)
        self.spec = spec
        self.params.extend(global_options())

    @override
    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        format_parameters(self, ctx, formatter)

    @override
    def invoke(self, ctx: click.Context) -> object:
        invocation = require_invocation(ctx)
        plain = bool(ctx.meta.get(PLAIN_META, False))
        if plain and not explicit_page(ctx):
            raise InvalidInput(
                "--plain needs an explicitly selected page",
                hint="Pass --limit N or --cursor CURSOR together with --plain",
            )
        selected = select_format(
            self.spec.formats,
            json_flag=invocation.flags.json,
            plain_flag=plain,
            stream=self.spec.stream,
            stdout_isatty=invocation.streams.stdout.isatty(),
        )
        invocation.begin(selected)
        _reject_conflicting_stdin(self, ctx, invocation)
        result = super().invoke(ctx)
        if not self.spec.stream:
            invocation.succeeded()
        emit_result(invocation, self.spec, result)
        if self.spec.stream:
            invocation.succeeded()
        return None


class Group(click.Group):
    """A command group that accepts the global flags and creates `Command` objects."""

    command_class = Command

    def __init__(
        self,
        name: str | None = None,
        *,
        help: str | None = None,
        epilog: str | None = None,
        context_settings: MutableMapping[str, object] | None = None,
    ) -> None:
        super().__init__(name, help=help, epilog=epilog, context_settings=context_settings)
        self.params.extend(global_options())

    @override
    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        format_parameters(self, ctx, formatter)
        self.format_commands(ctx, formatter)


def _option_label(param: click.Parameter) -> str:
    """The long flag name from parser metadata, such as `--spec`."""
    if isinstance(param, click.Option) and param.opts:
        longs = [opt for opt in param.opts if opt.startswith("--")]
        return longs[0] if longs else param.opts[0]
    return param.name or "stdin"


def _stdin_selections(command: click.Command, ctx: click.Context) -> list[str]:
    """The labels of this command's own stdin-capable inputs whose value is `-`."""
    return [
        _option_label(param)
        for param in command.params
        if not is_global_flag(param)
        and isinstance(param.type, StdinOrPath)
        and ctx.params.get(param.name or "") == "-"
    ]


def _reject_conflicting_stdin(
    command: click.Command, ctx: click.Context, invocation: Invocation
) -> None:
    """I3b: reject two stdin selections before the callback can read the first one.

    Only what the command line says is consulted, so no setting is resolved and a command
    that never reads a setting still accepts the global flags (D6c). `Invocation.claim_stdin`
    stays as the backstop for an input that reaches stdin by any other route.
    """
    selected = _stdin_selections(command, ctx)
    if len(selected) < 2:
        return
    raise InvalidInput(
        f"stdin cannot supply both {selected[0]} and {selected[1]}",
        hint="Pass one of them as a file path",
    )


def explicit_page(ctx: click.Context) -> bool:
    """Whether the caller selected a page with `--limit` or `--cursor` (H5a)."""
    return any(given(ctx, name) for name in PAGE_PARAMETERS)


def given(ctx: click.Context, parameter: str) -> bool:
    """Whether `parameter` was supplied on the command line rather than defaulted."""
    return ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
