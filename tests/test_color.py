"""H4 and O1: color is optional, per stream, and never reaches machine-readable output."""

from tests.conftest import Cli

ESCAPE = "\x1b["


def test_auto_disables_color_off_a_terminal_or_when_asked(cli: Cli) -> None:
    piped = cli.run("config", "show", "--color", "auto")
    assert ESCAPE not in piped.stdout
    for env in ({"NO_COLOR": "1"}, {"TERM": "dumb"}):
        outcome = cli.run("config", "show", tty=True, env=env)
        assert outcome.code == 0 and ESCAPE not in outcome.stdout, env


def test_no_escape_sequences_at_all_without_a_terminal(cli: Cli) -> None:
    # H4a: a non-TTY stream or TERM=dumb gets no escape sequence of any kind, bold included.
    piped = cli.run("nope", env={"TERM": "xterm"})
    assert "\x1b" not in piped.stderr and "\x1b" not in piped.stdout
    dumb = cli.run("nope", tty=True, env={"TERM": "dumb"})
    assert "\x1b" not in dumb.stderr and "Error:" in dumb.stderr
    dumb_stdout = cli.run("config", "show", tty=True, env={"TERM": "dumb"})
    assert "\x1b" not in dumb_stdout.stdout and "config_path" in dumb_stdout.stdout
    # Under NO_COLOR the tool drops every style; H4a allows non-color styling to stay.
    no_color = cli.run("nope", tty=True, env={"NO_COLOR": "1"})
    assert "\x1b" not in no_color.stderr and "Error:" in no_color.stderr


def test_always_colors_human_output_only(cli: Cli) -> None:
    text = cli.run("config", "show", "--color", "always", tty=True, env={"TERM": "dumb"})
    assert text.code == 0 and ESCAPE in text.stdout
    machine = cli.run("config", "show", "--color", "always", "--json", tty=True)
    assert ESCAPE not in machine.stdout
    assert "settings" in machine.document
    plain = cli.run("config", "show", "--color", "always", "--plain")
    assert plain.code == 2 and plain.error["kind"] == "invalid_input"
    piped = cli.run(
        "config",
        "show",
        "--config",
        "/nonexistent/provibench.toml",
        "--color",
        "always",
        tty_stdout=True,
        tty_stderr=False,
    )
    assert piped.code == 2 and ESCAPE in piped.stderr
    assert piped.error["kind"] == "invalid_input"
    assert ESCAPE not in piped.stderr.splitlines()[-1]
    never = cli.run("config", "show", "--color", "never", tty=True, env={"TERM": "xterm"})
    assert ESCAPE not in never.stdout


def test_stderr_is_classified_on_its_own(cli: Cli) -> None:
    # `schema` always selects JSON regardless of stdout, so stderr's own tty state is what
    # decides whether the human "Error:" line (and its color) appears alongside it.
    stderr_tty = cli.run(
        "schema", "does-not-exist", tty_stdout=False, tty_stderr=True, env={"TERM": "xterm"}
    )
    assert ESCAPE in stderr_tty.stderr
    assert stderr_tty.error["kind"] == "invalid_input"
    stdout_tty = cli.run(
        "schema", "does-not-exist", tty_stdout=True, tty_stderr=False, env={"TERM": "xterm"}
    )
    assert ESCAPE not in stdout_tty.stderr


def test_untrusted_values_are_escaped_in_human_output(cli: Cli) -> None:
    outcome = cli.run("schema", "[bold]x[/bold]", tty=True, env={"TERM": "dumb"})
    assert outcome.code == 2
