"""`config show`: every setting, its effective value, and its source (I2c).

The command reads only the settings declaration. It never opens a network target, never
touches trace or run state, and never makes an outbound request.
"""

from collections.abc import Sequence

import click

from provibench.core.context import Invocation
from provibench.core.documents import Document, as_document, nullable_string, obj, string
from provibench.core.registry import Command, Group, require_invocation
from provibench.core.settings import Setting, Source
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

MASK = "***"


def config_group(declaration: Sequence[Setting]) -> Group:
    """The `config` group with its `show` command, described by `declaration`."""
    group = Group(name="config", help="Inspect the tool's configuration.")
    setting_schema = obj(
        {"value": nullable_string(), "source": string(enum=[source.value for source in Source])},
        required=["value", "source"],
    )
    spec = CommandSpec(
        effects=Effects.READ_ONLY,
        output=obj(
            {
                "settings": obj(
                    {s.name: setting_schema for s in declaration},
                    required=[s.name for s in declaration],
                )
            },
            required=["settings"],
        ),
        render=render_settings,
    )

    @click.pass_context
    def show(ctx: click.Context) -> Document:
        invocation = require_invocation(ctx)
        settings: Document = {}
        for setting in declaration:
            resolved = invocation.settings[setting.name]
            value = resolved.value
            if setting.secret and value is not None:
                value = MASK
            settings[setting.name] = {"value": value, "source": resolved.source.value}
        return {"settings": settings}

    group.add_command(
        Command(
            "show",
            spec=spec,
            callback=show,
            help="Show every setting with its effective value and the source that supplied it.",
        )
    )
    return group


def render_settings(invocation: Invocation, document: Document) -> None:
    """A three-column table: setting, value, source."""
    from rich import box
    from rich.markup import escape
    from rich.table import Table

    table = Table(box=box.SIMPLE, header_style="bold")
    for column in ("Setting", "Value", "Source"):
        table.add_column(column)
    settings = as_document(document.get("settings")) or {}
    for name, raw in settings.items():
        entry = as_document(raw) or {}
        value = entry.get("value")
        table.add_row(
            escape(escape_terminal_text(name)),
            escape(escape_terminal_text("-" if value is None else str(value))),
            escape(escape_terminal_text(str(entry.get("source")))),
        )
    invocation.stdout_console().print(table)
