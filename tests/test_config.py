"""Configuration (I2): precedence, sources, the TOML loader, and `config show`."""

from pathlib import Path

from provibench.core.documents import as_document
from tests.conftest import Cli


def _settings(cli: Cli, *args: str, **kwargs: object) -> dict[str, dict[str, object]]:
    outcome = cli.run("config", "show", *args, **kwargs)  # pyright: ignore[reportArgumentType]
    assert outcome.code == 0, outcome.stderr
    settings = as_document(outcome.document["settings"]) or {}
    return {name: as_document(entry) or {} for name, entry in settings.items()}


def test_show_lists_every_setting_with_defaults(cli: Cli) -> None:
    settings = _settings(cli)
    assert set(settings) == {"config_path", "targets_path", "traces_dir", "runs_dir"}
    assert settings["config_path"] == {"value": None, "source": "default"}
    assert settings["targets_path"]["source"] == "default"
    assert str(settings["targets_path"]["value"]).endswith(".config/provibench/targets.toml")
    assert settings["traces_dir"] == {"value": str(cli.root / "traces"), "source": "default"}
    assert settings["runs_dir"] == {"value": str(cli.root / "runs"), "source": "default"}


def test_precedence_flag_over_env_over_file_over_default(cli: Cli, tmp_path: Path) -> None:
    config = tmp_path / "conf" / "provibench.toml"
    config.parent.mkdir()
    config.write_text(
        'traces_dir = "data/traces"\nruns_dir = "data/runs"\ntargets_path = "targets.toml"\n'
    )
    settings = _settings(
        cli, "--config", str(config), env={"PROVIBENCH_TRACES_DIR": "/tmp/env-traces"}
    )
    assert settings["config_path"] == {"value": str(config), "source": "flag"}
    assert settings["traces_dir"] == {"value": "/tmp/env-traces", "source": "env"}
    assert settings["runs_dir"] == {
        "value": str(config.parent / "data/runs"),
        "source": "config",
    }
    assert settings["targets_path"] == {
        "value": str(config.parent / "targets.toml"),
        "source": "config",
    }
    flagged = _settings(cli, "--targets", "/tmp/flagged.toml")
    assert flagged["targets_path"] == {"value": "/tmp/flagged.toml", "source": "flag"}


def test_default_config_file_is_used_when_present(cli: Cli) -> None:
    config = cli.home / ".config" / "provibench" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('traces_dir = "t"\n')
    settings = _settings(cli)
    assert settings["config_path"] == {"value": str(config), "source": "default"}
    assert settings["traces_dir"]["value"] == str(config.parent / "t")
    xdg = cli.root / "xdg" / "provibench" / "config.toml"
    xdg.parent.mkdir(parents=True)
    xdg.write_text('traces_dir = "x"\n')
    settings = _settings(cli, env={"XDG_CONFIG_HOME": str(cli.root / "xdg")})
    assert settings["traces_dir"]["value"] == str(xdg.parent / "x")
    settings = _settings(cli, env={"PROVIBENCH_CONFIG": str(xdg)})
    assert settings["config_path"] == {"value": str(xdg), "source": "env"}


def test_empty_environment_variable_counts_as_unset(cli: Cli) -> None:
    settings = _settings(cli, env={"PROVIBENCH_TRACES_DIR": ""})
    assert settings["traces_dir"] == {"value": str(cli.root / "traces"), "source": "default"}


def test_bad_configuration_files_are_invalid_input(cli: Cli, tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.toml"
    unknown.write_text('colour = "red"\n')
    outcome = cli.run("config", "show", "--config", str(unknown))
    assert outcome.code == 2 and outcome.error["kind"] == "invalid_input"
    assert "colour" in str(outcome.error["message"]) and unknown.name in str(
        outcome.error["message"]
    )
    wrong_type = tmp_path / "type.toml"
    wrong_type.write_text("traces_dir = 3\n")
    assert cli.run("config", "show", "--config", str(wrong_type)).error["kind"] == "invalid_input"
    broken = tmp_path / "broken.toml"
    broken.write_text("= nope\n")
    assert cli.run("config", "show", "--config", str(broken)).error["kind"] == "invalid_input"
    missing = cli.run("config", "show", "--config", str(tmp_path / "none.toml"))
    assert missing.code == 2 and missing.error["kind"] == "invalid_input"


def test_show_never_touches_the_filesystem_beyond_the_config_file(cli: Cli, tmp_path: Path) -> None:
    outcome = cli.run(
        "config",
        "show",
        "--targets",
        str(tmp_path / "nowhere" / "targets.toml"),
    )
    assert outcome.code == 0
    assert not (tmp_path / "nowhere").exists()


def test_undeclared_variables_change_nothing(cli: Cli) -> None:
    settings = _settings(cli, env={"PROVIBENCH_COLOUR": "always", "PROVIBENCH_TARGETSX": "x"})
    assert settings["targets_path"]["source"] == "default"


def test_show_human_table(cli: Cli) -> None:
    outcome = cli.run("config", "show", tty=True, env={"TERM": "dumb"})
    assert outcome.code == 0
    assert "traces_dir" in outcome.stdout and "default" in outcome.stdout
