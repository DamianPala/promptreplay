"""Subprocess tests: streams, pipes, and cross-process introspection."""

import json
from pathlib import Path

import pytest

from tests.conftest import run_process


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return {"HOME": str(home)}


def test_help_and_version(env: dict[str, str]) -> None:
    root = run_process(["--help"], env=env)
    assert root.code == 0
    assert "promptreplay schema" in root.stdout and "--json" in root.stdout
    for path in (["config", "show"], ["completion"]):
        assert run_process([*path, "--help"], env=env).code == 0
    version = run_process(["--version"], env=env)
    assert version.code == 0 and version.stdout.strip()
    no_args = run_process([], env=env)
    assert no_args.code == 2 and no_args.error["kind"] == "invalid_input"


def test_schema_index_from_a_process_is_pure_json(env: dict[str, str]) -> None:
    outcome = run_process(["schema"], env=env)
    assert outcome.code == 0 and outcome.stderr == ""
    assert json.loads(outcome.stdout)["conformance"]["extensions"] == []


def test_inspection_never_opens_stdin(env: dict[str, str]) -> None:
    """schema, --help, --version, completion, and config show do not read stdin."""
    calls = (
        ["schema"],
        ["--help"],
        ["--version"],
        ["completion", "bash"],
        ["config", "show"],
        ["schema", "--targets", str(Path(env["HOME"]) / "none.toml")],
    )
    for args in calls:
        outcome = run_process(args, env=env, hold_stdin=True, timeout=2)
        assert outcome.code == 0, (args, outcome.stderr)
        assert outcome.stdout, args
