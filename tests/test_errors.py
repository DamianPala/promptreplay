"""F1-F5: every error kind this tool emits, its exit code, and the F3 envelope location."""

import os

from tests.conftest import Cli


def test_usage_errors_are_invalid_input_with_exit_2(cli: Cli) -> None:
    cases = (
        ("config", "show", "--nope"),
        ("config",),
        ("nope",),
        ("--color", "sometimes", "config", "show"),
    )
    for args in cases:
        outcome = cli.run(*args)
        assert outcome.code == 2, args
        assert outcome.error["kind"] == "invalid_input", args
        assert outcome.stdout == "", args
    with_json_late = cli.run("config", "show", "--nope", "--json", tty=True)
    assert with_json_late.code == 2 and with_json_late.error["kind"] == "invalid_input"
    at_tty = cli.run("config", "show", "--nope", tty=True)
    assert at_tty.code == 2 and "Error:" in at_tty.stderr and "{" not in at_tty.stderr


def test_json_before_the_double_dash_selects_the_object_for_early_usage_errors(cli: Cli) -> None:
    root_level = cli.run("--bogus", "--json", tty=True)
    assert root_level.code == 2 and root_level.error["kind"] == "invalid_input"
    assert root_level.stdout == ""
    after_double_dash = cli.run("config", "show", "--nope", "--", "--json", tty=True)
    assert after_double_dash.code == 2 and "{" not in after_double_dash.stderr
    piped = cli.run("config", "show", "--nope", "--", "--json")
    assert piped.error["kind"] == "invalid_input"


def test_json_before_and_after_the_path(cli: Cli) -> None:
    before = cli.run("--json", "config", "show", tty=True)
    after = cli.run("config", "show", "--json", tty=True)
    assert before.code == 0 and after.code == 0
    assert before.document == after.document
    assert before.stdout == after.stdout  # same flag, same position in the terminal, byte-for-byte


def test_error_object_is_the_last_stderr_line_and_never_on_stdout(cli: Cli) -> None:
    outcome = cli.run("-v", "nope")
    assert outcome.code == 2
    assert outcome.stdout == ""
    assert outcome.error["kind"] == "invalid_input"
    machine = cli.run("nope", "--json")
    assert [line for line in machine.stderr.splitlines() if line] == [machine.stderr.strip()]
    assert machine.error["kind"] == "invalid_input"
    quiet = cli.run("-q", "-v", "nope")
    lines = [line for line in quiet.stderr.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith('{"error"')
    tty = cli.run("nope", tty=True)
    assert "{" not in tty.stderr and "Error:" in tty.stderr
    tty_stderr_only = cli.run("nope", tty_stdout=True, tty_stderr=False)
    assert tty_stderr_only.error["kind"] == "invalid_input"


def test_precondition_failed_names_force(cli: Cli) -> None:
    path = cli.home / ".local/share/bash-completion/completions/provibench"
    path.parent.mkdir(parents=True)
    path.write_text("# something else\n")
    outcome = cli.run("completion", "bash", "--install")
    assert outcome.code == 1 and outcome.error["kind"] == "precondition_failed"
    assert "--force" in str(outcome.error["hint"])
    assert path.read_text() == "# something else\n"
    forced = cli.run("completion", "bash", "--install", "--force", "--json")
    assert forced.code == 0 and forced.document["changed"] is True
    assert os.path.getsize(path) > 100
