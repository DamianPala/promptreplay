"""I1e relationships and runtime discovery exposed by schema descriptors."""

from promptreplay.core.documents import Document, as_document, as_list
from tests.conftest import Cli


def test_global_descriptions_expose_runtime_discovery(cli: Cli) -> None:
    index = cli.run("schema").document
    flags = {str(flag["name"]): flag for flag in _documents(index["global_flags"])}

    config = str(flags["config"]["description"])
    assert all(
        phrase in config
        for phrase in (
            "flag or environment selection",
            "replaces the default",
            "working directory",
            "When omitted",
            "config show",
            "paths and sources",
        )
    )
    targets = str(flags["targets"]["description"])
    assert all(
        phrase in targets
        for phrase in (
            "flag or environment selection",
            "working directory",
            "When omitted",
            "config show",
            "paths and sources",
        )
    )
    color = str(flags["color"]["description"])
    assert all(
        phrase in color for phrase in ("TTY", "TERM=dumb", "NO_COLOR", "`always`", "`never`")
    )


def test_command_descriptions_expose_relationships(cli: Cli) -> None:
    force = _flag(cli, "completion", "force")
    assert "content differs" in force
    install = _flag(cli, "completion", "install")
    assert "tool-owned completion path" in install
    shell = _argument(cli, "completion", "shell")
    assert "basename of $SHELL" in shell


def _flag(cli: Cli, path: str, name: str) -> str:
    detail = cli.run("schema", *path.split()).document
    flags = {str(flag["name"]): flag for flag in _documents(detail["flags"])}
    return str(flags[name]["description"])


def _argument(cli: Cli, path: str, name: str) -> str:
    detail = cli.run("schema", *path.split()).document
    arguments = {str(argument["name"]): argument for argument in _documents(detail["args"])}
    return str(arguments[name]["description"])


def _documents(value: object) -> list[Document]:
    return [document for item in as_list(value) or [] if (document := as_document(item))]
