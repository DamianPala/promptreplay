"""Composition root: the command tree, the settings declaration, and the process entry point.

Adding, replacing, or deleting a command group changes exactly one `add_command` line here.
"""

import os
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from promptreplay.commands.compare import compare as compare_command
from promptreplay.commands.endpoints import endpoints as endpoints_command
from promptreplay.commands.history import history as history_command
from promptreplay.commands.inspect import inspect as inspect_command
from promptreplay.commands.prices import prices as prices_command
from promptreplay.commands.probe import probe as probe_command
from promptreplay.commands.record import record as record_command
from promptreplay.commands.replay import replay as replay_command
from promptreplay.commands.report import report as report_command
from promptreplay.commands.scrub import scrub as scrub_command
from promptreplay.commands.sweep import sweep as sweep_command
from promptreplay.core.clock import SystemClock
from promptreplay.core.completion import completion_command
from promptreplay.core.config_command import config_group
from promptreplay.core.context import Invocation, Process, Streams
from promptreplay.core.entry import Root
from promptreplay.core.introspection import Conformance, schema_command
from promptreplay.core.settings import CwdPath, Setting, XdgPath, core_settings
from promptreplay.core.version import version_option

PROGRAM = "promptreplay"
CONFORMANCE = Conformance(
    name="cli-design-standard",
    standard="0.1.0-draft.7",
    extensions=(),
)


def packaged_targets_path() -> Path:
    """The packaged `targets.toml`'s real filesystem path, for the default nothing is at."""
    packaged = resources.files("promptreplay.data").joinpath("targets.toml")
    with resources.as_file(packaged) as path:
        return path


SETTINGS: list[Setting] = [
    Setting(
        "targets_path",
        "Targets file listing configured providers, models, and API key sources",
        flag="targets",
        env="PROMPTREPLAY_TARGETS",
        config_key="targets_path",
        default=XdgPath("XDG_CONFIG_HOME", ".config", "promptreplay/targets.toml"),
        is_path=True,
        packaged=packaged_targets_path,
    ),
    Setting(
        "traces_dir",
        "Directory holding recorded traces",
        env="PROMPTREPLAY_TRACES_DIR",
        config_key="traces_dir",
        default=CwdPath("traces"),
        is_path=True,
    ),
    Setting(
        "runs_dir",
        "Directory holding replay run results",
        env="PROMPTREPLAY_RUNS_DIR",
        config_key="runs_dir",
        default=CwdPath("runs"),
        is_path=True,
    ),
]
DECLARATION = [*core_settings(PROGRAM), *SETTINGS]

HELP = (
    "Record real agent-harness API traffic once, replay it byte-for-byte against LLM "
    "providers, and compare cache behaviour and cost per provider."
)
EPILOG = (
    f"Machine-readable interface: {PROGRAM} schema, then {PROGRAM} schema <command path>.\n\n"
    "Add --json to any command for JSON output; it is the default when stdout is not a terminal.\n"
    "A non-empty NO_INPUT disables prompts; use --yes for commands that require consent."
)


def build_cli() -> Root:
    """The root command with every group registered."""
    root = Root(
        name=PROGRAM,
        help=HELP,
        epilog=EPILOG,
        context_settings={"help_option_names": ["--help", "-h"]},
    )
    root.params.append(version_option(PROGRAM))
    root.add_command(schema_command(distribution=PROGRAM, conformance=CONFORMANCE))
    root.add_command(config_group(DECLARATION))
    root.add_command(completion_command(PROGRAM))
    root.add_command(history_command)
    root.add_command(compare_command)
    root.add_command(record_command)
    root.add_command(inspect_command)
    root.add_command(probe_command)
    root.add_command(sweep_command)
    root.add_command(replay_command)
    root.add_command(report_command)
    root.add_command(scrub_command)
    root.add_command(endpoints_command)
    root.add_command(prices_command)
    return root


def main(argv: Sequence[str], *, process: Process) -> int:
    """Run one invocation with injected boundaries and return its exit code."""
    invocation = Invocation(program=PROGRAM, process=process, declaration=DECLARATION)
    return build_cli().run(argv, invocation)


def run() -> None:
    """The console-script entry point."""
    process = Process(
        streams=Streams(sys.stdin, sys.stdout, sys.stderr),
        env=os.environ,
        clock=SystemClock(),
        cwd=Path.cwd(),
        home=Path.home(),
    )
    code = main(sys.argv[1:], process=process)
    sys.exit(code)
