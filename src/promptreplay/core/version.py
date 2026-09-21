"""The lazy `--version` (D4d): the package metadata is read only when asked for."""

import click

from promptreplay.core.registry import require_invocation


def tool_version(distribution: str) -> str:
    """The installed version of `distribution`, from `pyproject.toml` via the package metadata."""
    # Imported here: a module-level lookup would cost every invocation 15-25 ms.
    from importlib.metadata import version

    return version(distribution)


def version_option(distribution: str) -> click.Option:
    """An eager `--version` / `-V` flag that prints the version string and exits."""

    def show(ctx: click.Context, _param: click.Parameter, value: object) -> None:
        if not value or ctx.resilient_parsing:
            return
        stdout = require_invocation(ctx).streams.stdout
        stdout.write(f"{tool_version(distribution)}\n")
        stdout.flush()
        ctx.exit()

    return click.Option(
        ["--version", "-V"],
        is_flag=True,
        is_eager=True,
        expose_value=False,
        callback=show,
        help="Print the version and exit",
    )
