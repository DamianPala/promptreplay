"""`config show`: every setting, its effective value, and its source (I2c).

The command reads only the settings declaration. It never opens a network target, never
touches trace or run state, and never makes an outbound request.

`config init` copies the packaged file of the one setting that declares a `packaged`
fallback (`Setting.packaged`) to its effective user path, so `show`'s "packaged" source
has something in the user's own control to edit; it is added only when such a setting
exists in the declaration.
"""

from collections.abc import Sequence
from pathlib import Path

import click

from provibench.core.context import Invocation
from provibench.core.documents import Document, as_document, boolean, nullable_string, obj, string
from provibench.core.errors import PreconditionFailed
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
            value, source = _shown_value(setting, resolved.value, resolved.source)
            if setting.secret and value is not None:
                value = MASK
            settings[setting.name] = {"value": value, "source": source.value}
        return {"settings": settings}

    group.add_command(
        Command(
            "show",
            spec=spec,
            callback=show,
            help="Show every setting with its effective value and the source that supplied it.",
        )
    )
    packaged_setting = next((s for s in declaration if s.packaged is not None), None)
    if packaged_setting is not None:
        group.add_command(_init_command(packaged_setting))
    return group


def _init_command(setting: Setting) -> Command:
    """`config init`: copy `setting`'s packaged file to its effective user path."""
    packaged = setting.packaged
    if packaged is None:
        raise RuntimeError(f"{setting.name} declares no packaged fallback for `config init`")
    noun = setting.name.removesuffix("_path")
    output = obj({"path": string(), "changed": boolean()}, required=["path", "changed"])
    spec = CommandSpec(
        effects=Effects.IDEMPOTENT,
        output=output,
        output_description="path is the file that was written; changed is always true.",
        render=render_init,
    )

    @click.command(
        "init",
        cls=Command,
        spec=spec,
        help=f"Copy the packaged {noun} file to its effective user path, so it can be edited.\n\n"
        "Without --force, an existing file at that path is left untouched and the command "
        "refuses.",
    )
    @click.option("--force", is_flag=True, help="Overwrite an existing file at the effective path")
    @click.pass_context
    def init(ctx: click.Context, *, force: bool) -> Document:
        invocation = require_invocation(ctx)
        target = Path(invocation.setting(setting.name) or "")
        if target.exists() and not force:
            raise PreconditionFailed(
                f"{target} already exists", hint="Pass --force to overwrite it"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(packaged().read_bytes())
        return {"path": str(target), "changed": True}

    return init


def render_init(invocation: Invocation, document: Document) -> None:
    """Text output: the path the packaged file was copied to."""
    stdout = invocation.streams.stdout
    stdout.write(f"{escape_terminal_text(str(document.get('path')))}\n")
    stdout.flush()


def _shown_value(setting: Setting, value: str | None, source: Source) -> tuple[str | None, Source]:
    """`config show`'s own view of a path setting's default: the packaged file it names when
    the resolved default is not one that exists.

    This never changes what the setting resolves to elsewhere (`invocation.setting` still
    returns the plain, possibly nonexistent, default): only the display is different, so a
    command that treats a missing file as "nothing configured yet" keeps doing that.
    """
    if setting.packaged is None or source is not Source.DEFAULT:
        return value, source
    if value is not None and Path(value).is_file():
        return value, source
    return str(setting.packaged()), Source.PACKAGED


_SETTINGS_COLUMNS = ("setting", "value", "source")
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


def render_settings(invocation: Invocation, document: Document) -> None:
    """A three-column table: setting, value, source. The header is bold under H4 color."""
    from provibench.core.text_table import text_table

    settings = as_document(document.get("settings")) or {}
    rows = [_setting_cells(name, raw) for name, raw in settings.items()]
    lines = [escape_terminal_text(line) for line in text_table(_SETTINGS_COLUMNS, rows)]
    stdout = invocation.streams.stdout
    if lines and invocation.color_on(stdout):
        lines[0] = f"{_BOLD}{lines[0]}{_RESET}"
    stdout.write("\n".join(lines) + "\n")
    stdout.flush()


def _setting_cells(name: str, raw: object) -> list[str]:
    entry = as_document(raw) or {}
    value = entry.get("value")
    return [name, "-" if value is None else str(value), str(entry.get("source"))]
