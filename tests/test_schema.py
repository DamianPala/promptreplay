"""D6-D9: the introspection index and details agree with the parser and need nothing."""

import json
import re
from pathlib import Path

import click
import pytest

from provibench.app import build_cli
from provibench.bench.openrouter import Endpoint
from provibench.bench.probe import ProbeOptions, ProbeResult, ProbeRun
from provibench.bench.replay import ReplayOptions, ReplayResult, write_run
from provibench.bench.targets import Prices, RunSpec, Target
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage, append_entry
from provibench.core.documents import Document, as_document, as_list
from provibench.core.introspection import command_flags, listed_commands
from provibench.core.registry import Command
from tests.conftest import Cli

EXPECTED_COMMANDS = {
    "completion": ("idempotent", False, False),
    "config show": ("read_only", False, False),
    "endpoints": ("read_only", False, False),
    "inspect": ("read_only", False, False),
    "probe": ("non_idempotent", True, False),
    "record": ("non_idempotent", False, False),
    "replay": ("non_idempotent", True, False),
    "report": ("idempotent", False, False),
}


def _entries(index: Document) -> list[Document]:
    return [entry for entry in map(as_document, as_list(index["commands"]) or []) if entry]


def test_index_shape_and_conformance(cli: Cli) -> None:
    index = cli.run("schema").document
    assert index["schema_version"] == "1"
    assert index["tool_version"] == cli.run("--version").stdout.strip()
    assert index["format_defaults"] == {"tty": "text", "non_tty": "json"}
    assert index["exit_codes"] == {"0": "success", "1": "failure", "2": "usage error"}
    assert index["conformance"] == {
        "name": "cli-design-standard",
        "standard": "0.1.0-draft.7",
        "extensions": [],
    }
    names = [str(entry["name"]) for entry in _entries(index)]
    assert names == sorted(names)
    assert set(names) == set(EXPECTED_COMMANDS)
    assert "schema" not in names
    flags = {
        str(flag["name"]) for flag in map(as_document, as_list(index["global_flags"]) or []) if flag
    }
    assert flags == {"json", "quiet", "verbose", "color", "config", "targets"}


def test_every_entry_and_detail_carry_the_declared_metadata(cli: Cli) -> None:
    index = cli.run("schema").document
    for entry in _entries(index):
        name = str(entry["name"])
        effects, confirm, interactive = EXPECTED_COMMANDS[name]
        assert entry["effects"] == effects
        assert set(entry) == {"name", "description", "effects"}
        detail = cli.run("schema", *name.split()).document
        assert detail["name"] == name
        assert detail["effects"] == effects
        assert detail["confirm"] is confirm
        assert detail["interactive"] is interactive
        assert detail["description"] == entry["description"]
        if effects != "read_only":
            output = as_document(detail["output"]) or {}
            assert "changed" in (as_list(output["required"]) or [])


def test_details_match_the_runtime_parser(cli: Cli) -> None:
    root = build_cli()
    global_flag_names = {
        str(flag["name"])
        for flag in map(as_document, as_list(cli.run("schema").document["global_flags"]) or [])
        if flag
    }
    for path, command in listed_commands(root):
        detail = cli.run("schema", *path.split()).document
        flags = {
            str(flag["name"]): flag
            for flag in map(as_document, as_list(detail["flags"]) or [])
            if flag
        }
        assert set(flags) == {_descriptor_name(option) for option in command_flags(command)}
        assert not set(flags) & global_flag_names
        for option in command_flags(command):
            descriptor = flags[_descriptor_name(option)]
            assert descriptor["required"] is option.required
            assert descriptor["type"] == _expected_type(option)
            if isinstance(option.type, click.Choice):
                assert descriptor["enum"] == [str(choice) for choice in option.type.choices]
            info: dict[str, object] = option.to_info_dict()
            if option.is_flag:
                assert descriptor["default"] is bool(info["default"])
            elif info["default"] is None or option.required:
                assert "default" not in descriptor
            else:
                assert descriptor["default"] == info["default"]
        args = [arg for arg in map(as_document, as_list(detail["args"]) or []) if arg]
        parser_args = [p for p in command.params if isinstance(p, click.Argument)]
        assert [str(arg["name"]) for arg in args] == [p.name for p in parser_args]
        for descriptor, argument in zip(args, parser_args, strict=True):
            assert descriptor["required"] is argument.required
            assert descriptor["description"] == (argument.help or "")


def _descriptor_name(option: click.Option) -> str:
    long_names = [opt.lstrip("-") for opt in option.opts if opt.startswith("--")]
    return long_names[0] if long_names else option.opts[0].lstrip("-")


def _expected_type(option: click.Option) -> str:
    if option.is_flag or isinstance(option.type, click.types.BoolParamType):
        return "boolean"
    if isinstance(option.type, click.types.IntParamType):
        return "integer"
    if isinstance(option.type, click.types.FloatParamType):
        return "number"
    return "string"


def test_specific_descriptors(cli: Cli) -> None:
    completion = cli.run("schema", "completion").document
    assert completion["format_defaults"] == {"tty": "text", "non_tty": "text"}
    args = {str(a["name"]): a for a in map(as_document, as_list(completion["args"]) or []) if a}
    assert args["shell"]["required"] is False
    assert args["shell"]["enum"] == ["bash", "zsh", "fish"]
    flags = {str(f["name"]): f for f in map(as_document, as_list(completion["flags"]) or []) if f}
    assert flags["install"]["type"] == "boolean"
    show = cli.run("schema", "config", "show").document
    show_output = as_document(show["output"]) or {}
    assert "settings" in set(as_list(show_output["required"]) or [])


def test_index_lists_no_group_prefixes_and_no_help_or_version_flags(cli: Cli) -> None:
    index = cli.run("schema").document
    names = [str(entry["name"]) for entry in _entries(index)]
    assert "config" not in names
    flags = [
        str(flag["name"]) for flag in map(as_document, as_list(index["global_flags"]) or []) if flag
    ]
    assert "help" not in flags and "version" not in flags


def test_schema_accepts_global_flags_without_applying_them(cli: Cli, tmp_path: Path) -> None:
    outcome = cli.run("schema", "--config", str(tmp_path / "nonexistent.toml"), "-v", "-q")
    assert outcome.code == 0 and outcome.stderr == ""
    assert json.loads(outcome.stdout)["conformance"]["standard"] == "0.1.0-draft.7"
    detail = cli.run("schema", "config", "show", "--targets", str(tmp_path / "none.toml"))
    assert detail.code == 0 and detail.document["name"] == "config show"


def test_output_schemas_use_only_the_o4_subset(cli: Cli) -> None:
    allowed = {"type", "enum", "properties", "required", "items"}

    def check(schema: Document) -> None:
        assert set(schema) <= allowed, schema
        assert "type" in schema
        if schema["type"] == "object":
            properties = as_document(schema["properties"]) or {}
            required = as_list(schema["required"]) or []
            assert set(map(str, required)) <= set(properties)
            for child in properties.values():
                check(as_document(child) or {})
        if schema["type"] == "array":
            check(as_document(schema["items"]) or {})

    for entry in _entries(cli.run("schema").document):
        detail = cli.run("schema", *str(entry["name"]).split()).document
        check(as_document(detail["output"]) or {})


_O4_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
}

# Extra argv each listed command needs so it can succeed.
_SUCCESS_ARGV: dict[str, list[str]] = {
    "completion": ["bash"],
    "endpoints": ["deepseek/model"],
    "inspect": ["fixture"],
    "probe": ["probe-fixture", "t:model-a", "--yes"],
    "record": ["--name", "record-fixture", "--upstream", "https://example.invalid", "--append"],
    "replay": ["fixture", "--run", "t:model-a", "--yes"],
    "report": ["fixture-run"],
}


def _seed_traces(traces_dir: Path) -> None:
    """Two recorded traces: one single turn, and one with a rung and its warm turn."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    append_entry(
        traces_dir / "fixture.jsonl",
        TraceEntry(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            path="/v1/messages",
            body={"messages": [{"role": "user", "content": "hi"}]},
            conversation="c1",
        ),
    )
    for seq in (1, 2):
        append_entry(
            traces_dir / "probe-fixture.jsonl",
            TraceEntry(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                path="/v1/messages",
                body={
                    "model": "orig",
                    "system": [{"type": "text", "text": "You are helpful."}],
                    "messages": [{"role": "user", "content": f"question {seq}"}],
                    "max_tokens": 1024,
                },
                conversation="c1",
                response=RecordedResponse(
                    status=200, latency_ms=1.0, usage=Usage(input_tokens=100)
                ),
            ),
        )


def _seed_domain_fixtures(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture state and boundary fakes so record/inspect/probe/replay/report/endpoints succeed.

    No network, no API keys: `serve`, `run_probe`, `replay_all`, and `fetch_endpoints` are
    faked at their bench-module source (the point commands import them from, lazily, on
    every call), matching the "mock only boundaries" rule.
    """
    _seed_traces(cli.root / "traces")

    targets_path = cli.home / ".config" / "provibench" / "targets.toml"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.write_text(
        '[targets.t]\nurl = "https://x.test/v1/messages"\napi_key_env = "X_KEY"\n'
        'kind = "anthropic"\n'
    )

    target = Target(
        name="t", url="https://x.test/v1/messages", api_key_env="X_KEY", kind="anthropic"
    )
    spec = RunSpec(target=target, model="model-a")
    result = ReplayResult(
        seq=1,
        turn=1,
        status=200,
        latency_ms=1.0,
        message_id="m1",
        model="model-a",
        prompt_total=10,
        cached=0,
        cache_write=0,
        output_tokens=1,
    )
    write_run(
        cli.root / "runs",
        "fixture-run",
        "c1",
        [spec],
        ReplayOptions(),
        results={spec.label: [result]},
    )

    def fake_serve(upstream: str, trace_path: Path, host: str, port: int, **_: object) -> None:
        append_entry(
            trace_path,
            TraceEntry(
                seq=1,
                ts="2026-01-01T00:00:00Z",
                path="/v1/messages",
                body={"messages": [{"role": "user", "content": "hi"}]},
                conversation="c1",
            ),
        )

    async def fake_run_probe(
        specs: list[RunSpec],
        entries: list[TraceEntry],
        opts: ProbeOptions,
        env: object,
        **_: object,
    ) -> ProbeRun:
        """A served cold write and one served warm read per spec, so nothing is partial."""
        records: dict[str, list[ProbeResult]] = {}
        for spec in specs:
            cold = ProbeResult(
                spec_label=spec.label,
                rung=1,
                role="cold",
                attempt=0,
                seq=1,
                status=200,
                latency_ms=1.0,
                prompt_total=100,
            )
            warm = ProbeResult(
                spec_label=spec.label,
                rung=1,
                role="warm",
                attempt=1,
                seq=2,
                status=200,
                latency_ms=1.0,
                prompt_total=110,
                cached=100,
            )
            records[spec.label] = [cold, warm]
        return ProbeRun(run_hex="0123456789ab", options=opts.resolved(entries), records=records)

    async def fake_replay_all(
        specs: list[RunSpec], entries: object, opts: object, env: object, **kwargs: object
    ) -> dict[str, list[ReplayResult]]:
        on_progress = kwargs.get("on_progress")
        out: dict[str, list[ReplayResult]] = {}
        for s in specs:
            r = ReplayResult(
                seq=1,
                turn=1,
                status=200,
                latency_ms=1.0,
                message_id="m1",
                model=s.model,
                prompt_total=10,
                cached=0,
                cache_write=0,
                output_tokens=1,
            )
            if callable(on_progress):
                on_progress(r)
            out[s.label] = [r]
        return out

    async def fake_fetch_endpoints(client: object, model: str) -> list[Endpoint]:
        return [
            Endpoint(
                provider_name="Prov",
                tag="prov",
                prices=Prices(input=1.0, cache_read=0.0, cache_write=0.0, output=2.0),
            )
        ]

    monkeypatch.setattr("provibench.bench.proxy.serve", fake_serve)
    monkeypatch.setattr("provibench.bench.probe.run_probe", fake_run_probe)
    monkeypatch.setattr("provibench.bench.replay.replay_all", fake_replay_all)
    monkeypatch.setattr("provibench.bench.openrouter.fetch_endpoints", fake_fetch_endpoints)


def _check_o4(value: object, schema: Document) -> None:
    """Validate `value` against one O4 schema: type, enum, properties, required, items."""
    assert set(schema) <= {"type", "enum", "properties", "required", "items"}
    if not schema:
        return
    assert "type" in schema, schema
    kind = _o4_type(schema["type"], value)
    if kind is None:
        return
    assert type(value) in _O4_TYPES[kind]
    if "enum" in schema:
        assert value in (as_list(schema["enum"]) or [])
    _check_o4_children(value, schema, kind)


def _o4_type(declared: object, value: object) -> str | None:
    """The JSON type to check, or `None` when a nullable schema matched `null`."""
    if isinstance(declared, str):
        return declared
    kinds = as_list(declared)
    assert kinds is not None and len(kinds) == 2 and kinds[1] == "null"
    if value is None:
        return None
    first = kinds[0]
    assert isinstance(first, str)
    return first


def _check_o4_children(value: object, schema: Document, kind: str) -> None:
    """Required fields, declared properties, and homogeneous array items."""
    if kind == "object":
        document = as_document(value)
        assert document is not None
        properties = as_document(schema.get("properties")) or {}
        required = [str(item) for item in (as_list(schema.get("required")) or [])]
        assert set(required) <= set(document) <= set(properties)
        for key, child in document.items():
            _check_o4(child, as_document(properties[key]) or {})
    if kind == "array":
        items = as_document(schema.get("items")) or {}
        for child in as_list(value) or []:
            _check_o4(child, items)


def test_every_success_matches_its_output_schema_with_json_in_both_positions(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each listed command's real JSON document matches its own D8 `output` schema.

    `--json` is accepted before the command path and after it; both placements emit
    the same shape. Only the O4 keywords are honoured, so a declared extra keyword
    cannot hide a mismatch and an extra emitted field cannot hide inside one.
    """
    _seed_domain_fixtures(cli, monkeypatch)
    for entry in _entries(cli.run("schema").document):
        path = str(entry["name"])
        extra = _SUCCESS_ARGV.get(path, [])
        detail = cli.run("schema", *path.split()).document
        schema = as_document(detail["output"]) or {}
        for json_first in (True, False):
            args = [*path.split(), *extra]
            args.insert(0 if json_first else len(args), "--json")
            outcome = cli.run(*args)
            assert outcome.code == 0, (args, outcome.stderr)
            documents = outcome.records if detail.get("stream") else [outcome.document]
            for document in documents:
                _check_o4(document, schema)


def test_schema_needs_no_targets_file(cli: Cli, tmp_path: Path) -> None:
    outcome = cli.run("schema", env={"PROVIBENCH_TARGETS": str(tmp_path / "nowhere.toml")})
    assert outcome.code == 0
    assert json.loads(outcome.stdout)["schema_version"] == "1"


def test_unknown_path_is_a_usage_error_with_nearest_paths(cli: Cli) -> None:
    outcome = cli.run("schema", "config", "shwo")
    assert outcome.code == 2
    error = outcome.error
    assert error["kind"] == "invalid_input"
    assert "config show" in str(error["hint"])
    assert cli.run("schema", "schema").code == 2


def test_version_is_the_bare_version_string(cli: Cli) -> None:
    outcome = cli.run("-V")
    assert outcome.code == 0
    assert outcome.stdout == cli.run("--version").stdout
    version = cli.run("schema").document["tool_version"]
    assert outcome.stdout == f"{version}\n"
    assert re.fullmatch(r"\d+\.\d+\.\d+([-+.][0-9A-Za-z.]+)*", str(version)), version


def test_every_listed_command_is_a_registered_command() -> None:
    for _, command in listed_commands(build_cli()):
        assert isinstance(command, Command)


def test_help_describes_positionals_and_separates_global_flags(
    cli: Cli, capsys: pytest.CaptureFixture[str]
) -> None:
    # click writes help through its own echo, so it lands on the captured sys.stdout.
    for path in ("completion", "config show", "config"):
        assert cli.run(*path.split(), "--help").code == 0
        text = capsys.readouterr().out
        assert "Global options (before or after the command path):" in text
        assert "--json" in text.split("Global options")[1]
        assert "--json" not in text.split("Global options")[0]
    completion = cli.run("completion", "--help")
    assert completion.code == 0
    text = capsys.readouterr().out
    assert (
        "Positional arguments:" in text
        and "Shell to generate for; defaults to the basename of $SHELL" in text
    )
    assert text.count("SHELL") == 3
