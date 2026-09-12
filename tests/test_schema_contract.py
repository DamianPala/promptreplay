"""D7, D8, and I1 field-table checks independent of Click parser metadata."""

from copy import deepcopy
from typing import cast

import pytest

from provibench.core.documents import Document
from tests.conftest import Cli

_D7 = {
    "schema_version",
    "tool_version",
    "global_flags",
    "format_defaults",
    "exit_codes",
    "conformance",
    "commands",
}
_D8 = {"name", "description", "args", "flags", "effects", "confirm", "interactive"}
_INPUT = {"name", "description", "type", "required"}
_INVOCATION = {
    "args",
    "flags",
    "confirm",
    "interactive",
    "output",
    "stream",
    "delegates_stdout",
    "exit_codes",
    "format_defaults",
    "reserved_overrides",
}
_SCALAR_TYPES = {"string": {str}, "integer": {int}, "number": {int, float}, "boolean": {bool}}


def test_d7_index_matches_the_field_table(cli: Cli) -> None:
    _assert_index(cli.run("schema").document)


def test_d8_details_match_the_current_conditional_contract(cli: Cli) -> None:
    index = cli.run("schema").document
    global_names = {str(flag["name"]) for flag in _docs(index["global_flags"])}
    for entry in _docs(index["commands"]):
        path = str(entry["name"])
        _assert_detail(cli.run("schema", *path.split()).document, path, global_names)


def test_contract_checks_reject_known_table_violations(cli: Cli) -> None:
    """The checks fail on malformed documents, not only on parser disagreement."""
    index = cli.run("schema").document
    missing_d7 = deepcopy(index)
    missing_d7.pop("schema_version")
    with pytest.raises(AssertionError, match="D7"):
        _assert_index(missing_d7)

    invalid_exit_codes = deepcopy(index)
    invalid_exit_codes["exit_codes"] = ["0", "1", "2"]
    with pytest.raises(AssertionError):
        _assert_index(invalid_exit_codes)

    detail = cli.run("schema", "config", "show").document
    missing_output = deepcopy(detail)
    missing_output.pop("output")
    with pytest.raises(AssertionError, match="output"):
        _assert_detail(missing_output, "config show", set())

    duplicate_global = deepcopy(detail)
    duplicate_global["flags"] = [
        *_docs(duplicate_global["flags"]),
        deepcopy(_docs(index["global_flags"])[0]),
    ]
    with pytest.raises(AssertionError, match="global"):
        _assert_detail(duplicate_global, "config show", {"json"})

    completion = cli.run("schema", "completion").document
    invalid_aliases = deepcopy(completion)
    alias_flags = _docs(invalid_aliases["flags"])
    alias_flags[0]["aliases"] = "l"
    invalid_aliases["flags"] = alias_flags
    with pytest.raises(AssertionError):
        _assert_detail(invalid_aliases, "completion", set())

    invalid_default = deepcopy(completion)
    flags = _docs(invalid_default["flags"])
    flags[0]["default"] = [5]
    invalid_default["flags"] = flags
    with pytest.raises(AssertionError, match="array"):
        _assert_detail(invalid_default, "completion", set())

    invalid_stdin = deepcopy(completion)
    stdin_flags = _docs(invalid_stdin["flags"])
    stdin_flags[0]["accepts_stdin"] = "yes"
    invalid_stdin["flags"] = stdin_flags
    with pytest.raises(AssertionError):
        _assert_detail(invalid_stdin, "completion", set())


def _assert_index(index: Document) -> None:
    assert set(index) >= _D7, "D7 index is missing a required field"
    assert isinstance(index["schema_version"], str) and index["schema_version"].isdecimal()
    assert int(str(index["schema_version"])) > 0
    assert isinstance(index["tool_version"], str) and index["tool_version"]
    _assert_formats(index["format_defaults"])
    exit_codes = index["exit_codes"]
    assert isinstance(exit_codes, dict)
    assert {"0", "1", "2"} <= set(cast(dict[str, object], exit_codes))
    conformance_value = index["conformance"]
    assert isinstance(conformance_value, dict)
    conformance = cast(Document, conformance_value)
    assert {"name", "standard", "extensions"} <= set(conformance)
    assert conformance["name"] == "cli-design-standard"
    assert isinstance(conformance["standard"], str) and conformance["standard"]
    extensions = conformance["extensions"]
    assert isinstance(extensions, list)
    extension_names = cast(list[object], extensions)
    assert all(isinstance(name, str) and name for name in extension_names)
    assert len(extension_names) == len(set(extension_names))
    _assert_inputs(_docs(index["global_flags"]), flags=True)

    entries = _docs(index["commands"])
    names = [str(entry["name"]) for entry in entries]
    assert names == sorted(names) and len(names) == len(set(names))
    for entry in entries:
        assert {"name", "description", "effects"} <= set(entry)
        assert not set(entry) & _INVOCATION
        assert isinstance(entry["name"], str) and entry["name"] == " ".join(entry["name"].split())
        assert isinstance(entry["description"], str) and "\n" not in entry["description"]
        assert entry["effects"] in {"read_only", "idempotent", "non_idempotent"}


def _assert_detail(detail: Document, path: str, global_names: set[str]) -> None:
    assert set(detail) >= _D8, "D8 detail is missing a required field"
    assert detail["name"] == path
    assert isinstance(detail["description"], str) and "\n" not in detail["description"]
    assert detail["effects"] in {"read_only", "idempotent", "non_idempotent"}
    assert type(detail["confirm"]) is bool and type(detail["interactive"]) is bool
    _assert_inputs(_docs(detail["args"]), flags=False)
    flags = _docs(detail["flags"])
    _assert_inputs(flags, flags=True)
    assert not {str(flag["name"]) for flag in flags} & global_names, "D8 repeats a global flag"

    assert "output" in detail and isinstance(detail["output"], dict), f"{path} must declare output"
    assert "stream" not in detail
    assert ("format_defaults" in detail) == (path == "completion")
    if "format_defaults" in detail:
        _assert_formats(detail["format_defaults"])


def _assert_inputs(descriptors: list[Document], *, flags: bool) -> None:
    for position, descriptor in enumerate(descriptors):
        assert set(descriptor) >= _INPUT
        assert isinstance(descriptor["name"], str) and not descriptor["name"].startswith("-")
        assert isinstance(descriptor["description"], str) and "\n" not in descriptor["description"]
        input_type = descriptor["type"]
        assert isinstance(input_type, str) and input_type in _SCALAR_TYPES
        assert type(descriptor["required"]) is bool
        _assert_default_and_enum(descriptor, input_type)
        if "aliases" in descriptor:
            aliases_value = descriptor["aliases"]
            assert isinstance(aliases_value, list)
            aliases = cast(list[object], aliases_value)
            assert flags and all(
                isinstance(alias, str) and alias and not alias.startswith("-") for alias in aliases
            )
        for field in ("variadic", "repeatable", "accepts_stdin"):
            if field in descriptor:
                assert type(descriptor[field]) is bool
        if descriptor.get("variadic") is True:
            assert not flags and position == len(descriptors) - 1
        if descriptor.get("repeatable") is True:
            assert flags


def _assert_default_and_enum(descriptor: Document, input_type: str) -> None:
    if descriptor["required"]:
        assert "default" not in descriptor, "required input has a default"
    if "default" in descriptor:
        default = descriptor["default"]
        repeated = descriptor.get("repeatable") is True or descriptor.get("variadic") is True
        if repeated:
            assert isinstance(default, list), "repeatable and variadic defaults must be arrays"
        if isinstance(default, list):
            assert repeated, "array defaults require repeatable or variadic input"
        values = cast(list[object], default) if isinstance(default, list) else [default]
        assert all(type(value) in _SCALAR_TYPES[input_type] for value in values)
    if "enum" in descriptor:
        enum = descriptor["enum"]
        assert isinstance(enum, list)
        enum_values = cast(list[object], enum)
        assert all(type(value) in _SCALAR_TYPES[input_type] for value in enum_values)


def _assert_formats(value: object) -> None:
    assert isinstance(value, dict)
    defaults = cast(Document, value)
    assert {"tty", "non_tty"} <= set(defaults)
    assert all(isinstance(defaults[key], str) and defaults[key] for key in ("tty", "non_tty"))


def _docs(value: object) -> list[Document]:
    assert isinstance(value, list)
    items = cast(list[object], value)
    assert all(isinstance(item, dict) for item in items)
    return cast(list[Document], items)
