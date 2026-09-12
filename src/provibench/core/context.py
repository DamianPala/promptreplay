"""One invocation's boundaries: process dependencies, global flags, and settings.

Commands reach the outside world only through the `Invocation` on the click context, so
tests inject every boundary and nothing reads `sys` or `os.environ` directly.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from provibench.core.clock import Clock
from provibench.core.color import ColorMode, color_enabled
from provibench.core.errors import InvalidInput
from provibench.core.logsetup import configure_logging
from provibench.core.output import Format
from provibench.core.settings import Resolved, Setting, resolve_settings
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from rich.console import Console


@dataclass(frozen=True, slots=True)
class Streams:
    """The three standard streams of one invocation."""

    stdin: TextIO
    stdout: TextIO
    stderr: TextIO


@dataclass(frozen=True, slots=True)
class Process:
    """The ambient process dependencies captured for one invocation."""

    streams: Streams
    env: Mapping[str, str]
    clock: Clock
    cwd: Path
    home: Path


@dataclass(slots=True)
class GlobalFlags:
    """Values of the D7 global flags, as given on the command line."""

    json: bool = False
    quiet: bool = False
    verbose: bool = False
    color: ColorMode = ColorMode.AUTO
    config: str | None = None
    targets: str | None = None

    def record(self, name: str, value: object) -> None:
        """Store one flag's explicit value by its parser name."""
        match name:
            case "json":
                self.json = bool(value)
            case "quiet":
                self.quiet = bool(value)
            case "verbose":
                self.verbose = bool(value)
            case "color":
                self.color = ColorMode(str(value))
            case "config":
                self.config = str(value)
            case "targets":
                self.targets = str(value)
            case _:
                raise KeyError(f"{name!r} is not a global flag")

    def setting_values(self) -> dict[str, str | None]:
        """The flag values that feed settings, keyed by canonical flag name."""
        return {"config": self.config, "targets": self.targets}


class Invocation:
    """Everything one command run needs from its environment."""

    def __init__(
        self,
        *,
        program: str,
        process: Process,
        declaration: Sequence[Setting],
    ) -> None:
        self.program = program
        self.process = process
        self.declaration = declaration
        self.argv: Sequence[str] = ()
        self.flags = GlobalFlags()
        self.format: Format | None = None
        self.log = logging.getLogger(program)
        self._settings: dict[str, Resolved] | None = None
        self._stdin_claim: str | None = None
        self._consoles: dict[str, Console] = {}
        self.on_success: list[Callable[[], None]] = []

    def succeeded(self) -> None:
        """Run the hooks registered for a command that completed; a hook may still fail it."""
        for hook in self.on_success:
            hook()

    @property
    def home(self) -> Path:
        """The user's explicitly injected home directory."""
        return self.process.home

    @property
    def streams(self) -> Streams:
        """The invocation's injected standard streams."""
        return self.process.streams

    @property
    def env(self) -> Mapping[str, str]:
        """The invocation's injected environment."""
        return self.process.env

    @property
    def clock(self) -> Clock:
        """The invocation's injected clock."""
        return self.process.clock

    @property
    def cwd(self) -> Path:
        """The invocation's injected working directory."""
        return self.process.cwd

    @property
    def settings(self) -> Mapping[str, Resolved]:
        """Every declared setting, resolved once on first use."""
        if self._settings is None:
            self._settings = resolve_settings(
                self.declaration,
                flags=self.flags.setting_values(),
                env=self.env,
                cwd=self.cwd,
                home=self.home,
            )
        return self._settings

    def setting(self, name: str) -> str | None:
        """The effective value of one setting."""
        return self.settings[name].value

    def declared(self, name: str) -> Setting:
        """The declaration of one setting."""
        return next(setting for setting in self.declaration if setting.name == name)

    def begin(self, selected: Format) -> None:
        """Fix the stdout format and wire logging once every global flag is parsed."""
        self.format = selected
        configure_logging(
            self.program,
            self.streams.stderr,
            quiet=self.flags.quiet,
            verbose=self.flags.verbose,
        )

    def claim_stdin(self, purpose: str) -> TextIO:
        """Reserve stdin for one input; a second claim is a usage error (I3b)."""
        if self._stdin_claim is not None:
            raise InvalidInput(
                f"stdin cannot supply both {self._stdin_claim} and {purpose}",
                hint="Pass one of them as a file path",
            )
        self._stdin_claim = purpose
        return self.streams.stdin

    @property
    def stdin_claimed(self) -> bool:
        """Whether an input already reads stdin, which rules out prompting (I5a)."""
        return self._stdin_claim is not None

    @property
    def machine_readable(self) -> bool:
        """Whether the selected or requested stdout format is machine-readable."""
        if self.format is not None:
            return self.format.machine_readable
        return self.flags.json or "--json" in _before_double_dash(self.argv)

    def interactive_context(self) -> bool:
        """TTY stdin, no explicit machine format, and no input ban define interactive context."""
        return self.streams.stdin.isatty() and not self.flags.json and not self.env.get("NO_INPUT")

    def can_prompt(self) -> bool:
        """Whether a prompt may be written: interactive context, TTY stderr, free stdin."""
        return (
            self.interactive_context() and self.streams.stderr.isatty() and not self.stdin_claimed
        )

    def can_start_session(self) -> bool:
        """Whether an interactive session such as an editor may start (I5b)."""
        return (
            self.interactive_context()
            and self.streams.stdout.isatty()
            and not self.machine_readable
        )

    def color_on(self, stream: TextIO) -> bool:
        """Whether `stream` may carry color under `--color` and H4."""
        return color_enabled(self.flags.color, isatty=stream.isatty(), env=self.env)

    def stdout_console(self) -> "Console":
        """A rich console on stdout for human-readable results; imported lazily."""
        return self._console("stdout", self.streams.stdout)

    def stderr_console(self) -> "Console":
        """A rich console on stderr for human-readable messages; imported lazily."""
        return self._console("stderr", self.streams.stderr)

    def message(self, text: str, *, style: str | None = None) -> None:
        """A human-readable line on stderr, silenced by `--quiet`."""
        if self.flags.quiet:
            return
        stderr = self.streams.stderr
        visible = escape_terminal_text(text)
        if self.color_on(stderr):
            from rich.markup import escape

            self.stderr_console().print(escape(visible), style=style)
        else:
            stderr.write(f"{visible}\n")
            stderr.flush()

    def _console(self, key: str, stream: TextIO) -> "Console":
        console = self._consoles.get(key)
        if console is None:
            # Imported here so the JSON path never pays for rich (about 60 ms).
            from rich.console import Console

            colored = self.color_on(stream)
            console = Console(
                file=stream,
                force_terminal=colored,
                color_system="standard" if colored else None,
                no_color=not colored,
                highlight=False,
                soft_wrap=True,
            )
            self._consoles[key] = console
        return console


def _before_double_dash(argv: Sequence[str]) -> Sequence[str]:
    """The arguments the parser reads as flags: everything before `--` (I6b)."""
    if "--" in argv:
        return argv[: argv.index("--")]
    return argv
