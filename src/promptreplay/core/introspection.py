"""Introspection (D6-D9): the index and command detail derived from the parser's objects.

Nothing here is hand-maintained per command: descriptors come from the click parameters
the parser dispatches with, and the rest from each command's `CommandSpec`.
"""

import difflib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import click

from promptreplay.core.documents import Document
from promptreplay.core.errors import InvalidInput
from promptreplay.core.output import (
    STREAM_FORMAT_DEFAULTS,
    TOOL_FORMAT_DEFAULTS,
    Format,
    FormatDefaults,
)
from promptreplay.core.params import StdinOrPath
from promptreplay.core.registry import Command, Group, is_global_flag, require_invocation
from promptreplay.core.spec import CommandSpec, Effects
from promptreplay.core.version import tool_version

SCHEMA_VERSION = "1"
EXIT_CODES = {"0": "success", "1": "failure", "2": "usage error"}
INTROSPECTION_COMMAND = "schema"


@dataclass(frozen=True, slots=True)
class Conformance:
    """What the tool claims in the D7 `conformance` object: the standard and its extensions."""

    name: str
    standard: str
    extensions: Sequence[str]

    def to_document(self) -> Document:
        """The D7 `conformance` object."""
        return {"name": self.name, "standard": self.standard, "extensions": list(self.extensions)}


def schema_command(*, distribution: str, conformance: Conformance) -> Command:
    """The `schema` command: the D7 index alone, or D8 detail for a command path."""

    @click.command(
        INTROSPECTION_COMMAND,
        cls=Command,
        spec=CommandSpec(
            effects=Effects.READ_ONLY,
            format_defaults=FormatDefaults(tty=Format.JSON, non_tty=Format.JSON),
            listed=False,
        ),
        help="Print the machine-readable interface: the index, or the detail of a command path.",
    )
    @click.argument("path", nargs=-1, help="Command path segments; omit them for the index")
    @click.pass_context
    def schema(ctx: click.Context, path: tuple[str, ...]) -> Document:
        root = ctx.find_root().command
        if not isinstance(root, Group):
            raise RuntimeError("schema needs a Group at the root")
        require_invocation(ctx)
        if not path:
            return build_index(
                root, version=tool_version(distribution), conformance=conformance.to_document()
            )
        name = " ".join(path)
        command = find_command(root, path)
        if command is None:
            raise InvalidInput(f"Unknown command path {name!r}", hint=_nearest_hint(root, name))
        return build_detail(command, name)

    return schema


def build_index(root: Group, *, version: str, conformance: Document) -> Document:
    """The D7 index of `root`."""
    ctx = _describing_context(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": version,
        "global_flags": [describe_option(option, ctx) for option in global_flags(root)],
        "format_defaults": TOOL_FORMAT_DEFAULTS.to_document(),
        "exit_codes": dict(EXIT_CODES),
        "conformance": conformance,
        "commands": [
            {
                "name": name,
                "description": description(command),
                "effects": command.spec.effects.value,
            }
            for name, command in sorted(listed_commands(root), key=lambda entry: entry[0])
        ],
    }


def build_detail(command: Command, name: str) -> Document:
    """The D8 detail of one command."""
    spec = command.spec
    ctx = _describing_context(command)
    detail: Document = {
        "name": name,
        "description": description(command),
        "args": [
            describe_argument(p, ctx) for p in command.params if isinstance(p, click.Argument)
        ],
        "flags": [describe_option(p, ctx) for p in command_flags(command)],
        "effects": spec.effects.value,
        "confirm": spec.confirm,
        "interactive": spec.interactive,
    }
    if spec.stream:
        detail["stream"] = True
    if spec.output is not None:
        detail["output"] = spec.output
    if spec.output_description is not None:
        detail["output_description"] = spec.output_description
    # O2a: a stream command's implied `ndjson` name is not a difference from the tool-wide
    # defaults, so only an explicit departure from the implied defaults is declared.
    implied = STREAM_FORMAT_DEFAULTS if spec.stream else TOOL_FORMAT_DEFAULTS
    if spec.formats != implied:
        detail["format_defaults"] = spec.formats.to_document()
    return detail


def listed_commands(group: click.Group, prefix: str = "") -> Iterator[tuple[str, Command]]:
    """Every dispatchable command under `group` with its full path, in registration order."""
    for name, command in group.commands.items():
        path = f"{prefix} {name}".strip()
        if isinstance(command, click.Group):
            yield from listed_commands(command, path)
        elif isinstance(command, Command) and command.spec.listed and not command.hidden:
            yield path, command


def find_command(root: click.Group, path: Sequence[str]) -> Command | None:
    """The listed command at `path`, or `None`."""
    name = " ".join(path)
    return next((command for entry, command in listed_commands(root) if entry == name), None)


def global_flags(root: Group) -> list[click.Option]:
    """The D7 global flags, taken from the root's own parameters."""
    return [p for p in root.params if isinstance(p, click.Option) and is_global_flag(p)]


def command_flags(command: click.Command) -> list[click.Option]:
    """A command's own flags: every visible option that is not a global flag."""
    return [
        p
        for p in command.params
        if isinstance(p, click.Option) and not is_global_flag(p) and not p.hidden
    ]


def description(command: click.Command) -> str:
    """The one-line purpose: the first line of the command's help."""
    lines = (command.help or "").strip().splitlines()
    return lines[0] if lines else ""


def _describing_context(command: click.Command) -> click.Context:
    """A context used only to read `command`'s declared defaults.

    It carries no `default_map`, so a parameter's default is the one the command declares
    and not whatever a running invocation happened to layer on top of it.
    """
    return click.Context(command)


def describe_argument(argument: click.Argument, ctx: click.Context) -> Document:
    """The I1 descriptor of a positional argument."""
    descriptor: Document = {
        "name": argument.name or "",
        "description": argument.help or "",
        "type": _type_name(argument),
        "required": argument.required,
    }
    _add_default(descriptor, argument, ctx)
    _add_enum(descriptor, argument)
    if argument.nargs == -1:
        descriptor["variadic"] = True
    return descriptor


def describe_option(option: click.Option, ctx: click.Context) -> Document:
    """The I1 descriptor of a flag."""
    names = [opt.lstrip("-") for opt in [*option.opts, *option.secondary_opts]]
    long_names = [opt.lstrip("-") for opt in option.opts if opt.startswith("--")]
    name = long_names[0] if long_names else names[0]
    descriptor: Document = {
        "name": name,
        "description": option.help or "",
        "type": _type_name(option),
        "required": option.required,
    }
    _add_default(descriptor, option, ctx)
    _add_enum(descriptor, option)
    aliases = [alias for alias in names if alias != name]
    if aliases:
        descriptor["aliases"] = aliases
    if option.multiple:
        descriptor["repeatable"] = True
    if isinstance(option.type, StdinOrPath):
        descriptor["accepts_stdin"] = True
    return descriptor


def _type_name(param: click.Parameter) -> str:
    if isinstance(param, click.Option) and param.is_flag:
        return "boolean"
    match param.type:
        case click.types.BoolParamType():
            return "boolean"
        case click.types.IntParamType():
            return "integer"
        case click.types.FloatParamType():
            return "number"
        case _:
            return "string"


def _add_default(descriptor: Document, param: click.Parameter, ctx: click.Context) -> None:
    """I1d: the built-in default when the parser supplies one; never for required inputs.

    `get_default` is what the parser itself calls, so the descriptor states the value a
    caller who omits the input actually gets. `call=False` leaves a callable default
    uncalled, and the type check then keeps it, and click's absence sentinel, out of the
    schema: neither has a JSON form, and a sentinel would read as a real default.
    """
    if param.required:
        return
    default: object = param.get_default(ctx, call=False)
    if isinstance(param, click.Option) and param.is_flag:
        descriptor["default"] = bool(default)
        return
    if isinstance(default, tuple):
        default = list(cast(tuple[object, ...], default))
    if isinstance(default, str | int | float | list):
        descriptor["default"] = default


def _add_enum(descriptor: Document, param: click.Parameter) -> None:
    if isinstance(param.type, click.Choice):
        descriptor["enum"] = [str(choice) for choice in param.type.choices]


def _nearest_hint(root: Group, name: str) -> str:
    names = [entry for entry, _ in listed_commands(root)]
    close = difflib.get_close_matches(name, names, n=3, cutoff=0.4)
    if close:
        return "Nearest command paths: " + ", ".join(close)
    return f"Run {INTROSPECTION_COMMAND} without arguments to list command paths"
