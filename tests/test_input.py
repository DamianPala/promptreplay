"""Bounded UTF-8 input mechanics shared by configuration and future command inputs."""

import io
from pathlib import Path

import pytest

from provibench.core.documents import as_document, as_list
from provibench.core.input import InputTooLarge, InvalidUtf8, read_bounded_file, read_bounded_stream
from provibench.core.settings import core_settings
from tests.conftest import Cli


@pytest.mark.parametrize("reader", ["file", "stream"])
def test_bounded_readers_measure_utf8_bytes_not_characters(tmp_path: Path, reader: str) -> None:
    content = "é" * 2
    path = tmp_path / "input"
    path.write_text(content, encoding="utf-8")

    if reader == "file":
        exact = read_bounded_file(path, max_bytes=4)
    else:
        exact = read_bounded_stream(io.StringIO(content), max_bytes=4)

    assert exact == content
    with pytest.raises(InputTooLarge):
        if reader == "file":
            read_bounded_file(path, max_bytes=3)
        else:
            read_bounded_stream(io.StringIO(content), max_bytes=3)


def test_bounded_file_rejects_invalid_utf8_and_preserves_os_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.write_bytes(b"\xff")

    with pytest.raises(InvalidUtf8):
        read_bounded_file(invalid, max_bytes=10)
    with pytest.raises(FileNotFoundError):
        read_bounded_file(tmp_path / "missing", max_bytes=10)


def test_bounded_stream_recovers_invalid_raw_bytes_without_exposing_them() -> None:
    stream = io.TextIOWrapper(io.BytesIO(b"\xff"), encoding="utf-8", errors="surrogateescape")

    with pytest.raises(InvalidUtf8):
        read_bounded_stream(stream, max_bytes=10)


@pytest.mark.parametrize("selector", ["flag", "environment", "default"])
def test_config_byte_bound_rejects_before_command_execution(cli: Cli, selector: str) -> None:
    path = (
        cli.root / "config.toml"
        if selector != "default"
        else cli.home / ".config/provibench/config.toml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    args: tuple[str, ...] = ("--config", str(path)) if selector == "flag" else ()
    env = {"PROVIBENCH_CONFIG": str(path)} if selector == "environment" else None
    path.write_text(_config_of_size(65_536), encoding="utf-8")
    assert cli.run("config", "show", *args, env=env).code == 0

    path.write_text(_config_of_size(65_537), encoding="utf-8")
    rejected = cli.run("config", "show", *args, env=env)

    assert rejected.code == 2 and rejected.error["kind"] == "invalid_input"

    path.write_bytes(b"\xff")
    invalid_utf8 = cli.run("config", "show", *args, env=env)
    assert invalid_utf8.code == 2 and invalid_utf8.error["kind"] == "invalid_input"


def test_schema_descriptions_disclose_all_bounded_inputs(cli: Cli) -> None:
    index = cli.run("schema").document
    config = _flag_description(index, "config")
    assert "65536 UTF-8 bytes" in config


def test_core_setting_descriptions_disclose_their_input_bounds() -> None:
    descriptions = {setting.name: setting.description for setting in core_settings("provibench")}

    assert "65536 UTF-8 bytes" in descriptions["config_path"]


def _config_of_size(size: int) -> str:
    prefix = 'traces_dir = "t"\n#'
    return prefix + "x" * (size - len(prefix))


def _flag_description(document: object, name: str) -> str:
    fields = as_document(document) or {}
    flags = (
        as_document(flag) for flag in as_list(fields.get("flags", fields.get("global_flags"))) or []
    )
    return str(
        next(flag for flag in flags if flag is not None and flag.get("name") == name)["description"]
    )
