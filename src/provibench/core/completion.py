"""Shell completions (H2): print or install click's completion script for bash, zsh, or fish.

The script comes from click's own completion classes, so the candidates always match the
installed interface. `--install` writes the tool-owned path for the shell and refuses to
overwrite foreign content unless `--force` overrides that precondition (R3d).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import click
from click.shell_completion import get_completion_class

from provibench.core.context import Invocation
from provibench.core.documents import Document, boolean, obj, string
from provibench.core.errors import CliError, InvalidInput, PreconditionFailed
from provibench.core.output import Format, FormatDefaults
from provibench.core.registry import Command, require_invocation
from provibench.core.settings import XdgPath
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

SHELLS = ("bash", "zsh", "fish")


class CompletionError(CliError):
    """The completion script could not be read or installed at its selected location."""

    kind = "completion_error"


def completion_variable(program: str) -> str:
    """The environment variable through which a shell asks `program` for candidates."""
    return f"_{program}_COMPLETE".upper().replace("-", "_").replace(".", "_")


@dataclass(frozen=True, slots=True)
class ShellTarget:
    """Where a shell's completion script lives and how the user activates it."""

    location: XdgPath
    activation: str


def shell_targets(program: str) -> Mapping[str, ShellTarget]:
    """The tool-owned completion path and activation note for each supported shell."""
    return {
        "bash": ShellTarget(
            XdgPath("XDG_DATA_HOME", ".local/share", f"bash-completion/completions/{program}"),
            "Restart bash, or source the file in the current shell.",
        ),
        "zsh": ShellTarget(
            XdgPath("XDG_DATA_HOME", ".local/share", f"zsh/site-functions/_{program}"),
            "Add the directory to fpath before compinit in ~/.zshrc, then restart zsh.",
        ),
        "fish": ShellTarget(
            XdgPath("XDG_CONFIG_HOME", ".config", f"fish/completions/{program}.fish"),
            "Restart fish; it loads the file automatically.",
        ),
    }


def completion_command(program: str) -> Command:
    """The `completion` command for `program`."""
    spec = CommandSpec(
        effects=Effects.IDEMPOTENT,
        output=obj(
            {
                "shell": string(enum=list(SHELLS)),
                "script": string(),
                "path": string(),
                "changed": boolean(),
            },
            required=["shell", "changed"],
        ),
        output_description=(
            "Without --install, the generated script is in script. With --install, path is "
            "the installed file and changed says whether it was written."
        ),
        format_defaults=FormatDefaults(tty=Format.TEXT, non_tty=Format.TEXT),
        render=render_completion,
    )

    @click.command(
        "completion",
        cls=Command,
        spec=spec,
        help="Print the shell completion script, or install it with --install.\n\n"
        "Without SHELL, the basename of $SHELL selects the shell. The result is the script itself,"
        " so text is the default format on every stdout.",
    )
    @click.argument(
        "shell",
        required=False,
        type=click.Choice(SHELLS),
        help="Shell to generate for; defaults to the basename of $SHELL",
    )
    @click.option(
        "--install",
        is_flag=True,
        help="Write the script to the tool-owned completion path for the shell",
    )
    @click.option(
        "--force",
        is_flag=True,
        help="With --install, overwrite a file whose content differs",
    )
    @click.pass_context
    def completion(ctx: click.Context, shell: str | None, install: bool, force: bool) -> Document:
        invocation = require_invocation(ctx)
        name = shell or _shell_from_env(invocation)
        script = completion_script(ctx.find_root().command, program, name)
        if not install:
            return {"shell": name, "script": script, "changed": False}
        target = shell_targets(program)[name]
        path = target.location.resolve(invocation.env, invocation.home)
        changed = _install(path, script, force=force)
        return {"shell": name, "path": str(path), "changed": changed}

    return completion


def completion_script(root: click.Command, program: str, shell: str) -> str:
    """The completion script click generates for `shell`."""
    completion_class = get_completion_class(shell)
    if completion_class is None:
        raise InvalidInput(
            f"Unsupported shell {shell!r}", hint=f"Choose one of {', '.join(SHELLS)}"
        )
    completer = completion_class(root, {}, program, completion_variable(program))
    # `source()` is click's public entry point and the only call site to revisit on a click
    # upgrade. Bash's override also runs `bash --version` and warns on stderr below 4.4.
    return completer.source()


def _shell_from_env(invocation: Invocation) -> str:
    value = invocation.env.get("SHELL")
    if not value:
        raise InvalidInput(
            "No shell given and SHELL is unset", hint="Pass the shell: bash, zsh, or fish"
        )
    name = Path(value).name
    if name not in SHELLS:
        raise InvalidInput(
            f"Unsupported shell {name!r} from SHELL", hint="Pass one of bash, zsh, or fish"
        )
    return name


def _install(path: Path, script: str, *, force: bool) -> bool:
    script_bytes = script.encode()
    try:
        if path.exists():
            with path.open("rb") as existing:
                content = existing.read(len(script_bytes) + 1)
            if content == script_bytes:
                return False
            if not force:
                raise PreconditionFailed(
                    f"{path} exists with different content",
                    hint="Repeat with --force to overwrite it",
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(script_bytes)
        return True
    except OSError as error:
        raise CompletionError(
            f"Cannot install completion at {path}: {error.strerror or error}",
            hint="Set the XDG data or config directory to a writable location",
            context={"path": str(path)},
        ) from error


def render_completion(invocation: Invocation, document: Document) -> None:
    """Text output: the script itself, or where it was installed and how to activate it."""
    stdout = invocation.streams.stdout
    script = document.get("script")
    if isinstance(script, str):
        stdout.write(script)
        if not script.endswith("\n"):
            stdout.write("\n")
        return
    shell = str(document.get("shell"))
    state = "Installed" if document.get("changed") else "Already installed"
    path = escape_terminal_text(str(document.get("path")))
    stdout.write(f"{state} {shell} completion at {path}\n")
    stdout.write(f"{shell_targets(invocation.program)[shell].activation}\n")
