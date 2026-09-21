"""The static agent skill uses only commands and flags the installed CLI exposes."""

import shlex
from pathlib import Path

import click

from promptreplay.app import build_cli
from promptreplay.core.introspection import command_flags, global_flags, listed_commands
from promptreplay.core.registry import Group

SKILL = Path(__file__).parents[1] / "skills/promptreplay/SKILL.md"
README = Path(__file__).parents[1] / "README.md"


def test_skill_commands_and_flags_match_introspection() -> None:
    root = build_cli()
    _assert_commands_match_introspection(root, _command_lines(SKILL))


def test_readme_commands_and_flags_match_introspection() -> None:
    root = build_cli()
    _assert_commands_match_introspection(root, _command_lines(README))


def _assert_commands_match_introspection(root: Group, lines: list[str]) -> None:
    commands = dict(listed_commands(root))
    shared_flags = {_long_name(option) for option in global_flags(root)}

    for line in lines:
        tokens = shlex.split(line)
        command_start = tokens.index("promptreplay")
        tokens = tokens[command_start:]
        assert tokens[0] == "promptreplay"
        command_name = " ".join(tokens[1:3])
        command_size = 2 if command_name in commands else 1
        command_name = " ".join(tokens[1 : 1 + command_size])
        assert command_name in commands, line
        command_flags_set = {_long_name(option) for option in command_flags(commands[command_name])}
        known_flags = shared_flags | command_flags_set
        for token in tokens[1 + command_size :]:
            if token.startswith("--"):
                flag = token[2:].split("=", 1)[0]
                assert flag in known_flags, f"{flag!r} in {line!r} is not in schema"


def _command_lines(path: Path) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif in_fence and "promptreplay " in line:
            lines.append(line[line.index("promptreplay ") :])
    return lines


def _long_name(option: click.Option) -> str:
    opts = option.opts
    long_names = [name[2:] for name in opts if name.startswith("--")]
    return long_names[0] if long_names else opts[0].lstrip("-")
