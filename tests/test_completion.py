"""H2 and R3d: completion scripts for three shells and the guarded install."""

from tests.conftest import Cli


def test_scripts_for_every_shell(cli: Cli) -> None:
    for shell, marker in (
        ("bash", "_PROMPTREPLAY_COMPLETE"),
        ("zsh", "#compdef promptreplay"),
        ("fish", "promptreplay"),
    ):
        outcome = cli.run("completion", shell)
        assert outcome.code == 0, shell
        assert marker in outcome.stdout
        assert not outcome.stdout.startswith("{")
    as_json = cli.run("completion", "bash", "--json").document
    assert as_json["shell"] == "bash" and as_json["changed"] is False
    assert "_PROMPTREPLAY_COMPLETE" in str(as_json["script"])


def test_shell_comes_from_the_environment(cli: Cli) -> None:
    assert "#compdef" in cli.run("completion", env={"SHELL": "/usr/bin/zsh"}).stdout
    missing = cli.run("completion", unset=("SHELL",))
    assert missing.code == 2 and missing.error["kind"] == "invalid_input"
    unsupported = cli.run("completion", env={"SHELL": "/bin/tcsh"})
    assert unsupported.code == 2
    assert cli.run("completion", "powershell").code == 2


def test_install_writes_the_tool_owned_path_and_reports_changed(cli: Cli) -> None:
    first = cli.run("completion", "fish", "--install", "--json").document
    path = cli.home / ".config" / "fish" / "completions" / "promptreplay.fish"
    assert first == {"shell": "fish", "path": str(path), "changed": True}
    assert path.read_text() == cli.run("completion", "fish").stdout
    second = cli.run("completion", "fish", "--install", "--json").document
    assert second["changed"] is False
    human = cli.run("completion", "zsh", "--install", tty=True, env={"TERM": "dumb"})
    zsh_path = cli.home / ".local" / "share" / "zsh" / "site-functions" / "_promptreplay"
    assert str(zsh_path) in human.stdout and "fpath" in human.stdout
    assert zsh_path.is_file()
    xdg = cli.run(
        "completion", "bash", "--install", "--json", env={"XDG_DATA_HOME": str(cli.root / "data")}
    ).document
    expected_path = cli.root / "data" / "bash-completion" / "completions" / "promptreplay"
    assert xdg["path"] == str(expected_path)
