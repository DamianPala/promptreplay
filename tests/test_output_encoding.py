"""O5a: `--json` output is always valid UTF-8, even when a path carries a byte that is not.

`os.fsdecode`-style decoding of an OS-supplied path (an environment variable, in practice)
uses `surrogateescape`, which puts an unpaired surrogate such as U+DCFF into the resulting
Python `str` for a byte the locale encoding cannot represent. Left alone, that surrogate
either fails to encode as UTF-8 outright or, on a stream that itself uses
`surrogateescape`, comes back out as the original invalid byte. Either way, `--json` stdout
is no longer valid UTF-8. `sanitize_document` (`provibench.core.documents`) replaces it with
U+FFFD at the output boundary; these tests build the same kind of string a decoded path
would produce, without touching the filesystem, and check the boundary from both ends:
the raw text must encode as UTF-8, and no string in the parsed document may contain a
lone surrogate.
"""

import json
from typing import cast

from provibench.core.documents import Document, as_document, sanitize_document
from tests.conftest import Cli

SURROGATE_PATH = "/tmp/provibench-test-x\udcff/targets.toml"
"""A path string shaped like `os.fsdecode(b"/tmp/provibench-test-x\\xff/targets.toml")`."""


def _assert_no_lone_surrogate(value: object) -> None:
    if isinstance(value, str):
        assert all(not 0xD800 <= ord(char) <= 0xDFFF for char in value), value
    elif isinstance(value, dict):
        for key, item in cast("dict[object, object]", value).items():
            _assert_no_lone_surrogate(key)
            _assert_no_lone_surrogate(item)
    elif isinstance(value, list):
        for item in cast("list[object]", value):
            _assert_no_lone_surrogate(item)


def _valid_utf8_document(stdout: str) -> Document:
    """`stdout` decoded strictly as UTF-8 bytes, then parsed as one JSON document."""
    document = as_document(json.loads(stdout.encode("utf-8").decode("utf-8")))
    assert document is not None, stdout
    return document


def test_config_show_json_sanitizes_a_surrogate_escaped_targets_path(cli: Cli) -> None:
    outcome = cli.run("config", "show", "--json", env={"PROVIBENCH_TARGETS": SURROGATE_PATH})
    assert outcome.code == 0, outcome.stderr
    document = _valid_utf8_document(outcome.stdout)
    _assert_no_lone_surrogate(document)
    settings = as_document(document["settings"]) or {}
    targets_path = as_document(settings["targets_path"]) or {}
    assert targets_path["source"] == "env"
    assert "\N{REPLACEMENT CHARACTER}" in str(targets_path["value"])


def test_config_show_human_table_also_sanitizes(cli: Cli) -> None:
    outcome = cli.run(
        "config", "show", tty=True, env={"TERM": "dumb", "PROVIBENCH_TARGETS": SURROGATE_PATH}
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout.encode("utf-8")  # must not raise
    _assert_no_lone_surrogate(outcome.stdout)


def test_json_error_context_message_with_surrogate_escaped_path_is_sanitized(cli: Cli) -> None:
    outcome = cli.run("config", "show", "--json", "--config", SURROGATE_PATH)
    assert outcome.code == 2
    assert outcome.stderr.encode("utf-8")  # must not raise
    error = _valid_utf8_document(outcome.stderr)
    _assert_no_lone_surrogate(error)


def test_sanitize_document_replaces_lone_surrogates_anywhere_in_the_tree() -> None:
    document: Document = {
        "value": SURROGATE_PATH,
        "nested": {"items": [SURROGATE_PATH, {"deep": SURROGATE_PATH}]},
        "clean": "ordinary text with é, é, and \U0001f600",
    }
    sanitized = sanitize_document(document)
    _assert_no_lone_surrogate(sanitized)
    assert json.dumps(sanitized, ensure_ascii=False).encode("utf-8")
    assert sanitized["clean"] == document["clean"]


def test_sanitize_document_leaves_ascii_and_valid_unicode_untouched() -> None:
    document: Document = {"name": "billing", "note": "café", "count": 3, "ok": True, "none": None}
    assert sanitize_document(document) == document


def test_json_document_is_indented_on_a_terminal(cli: Cli) -> None:
    """A JSON document read directly on a terminal is pretty-printed, not one long line."""
    outcome = cli.run("schema", "--json", tty_stdout=True)
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout.endswith("\n")
    assert outcome.stdout.count("\n") > 1
    assert json.loads(outcome.stdout) == outcome.document


def test_json_document_stays_one_line_off_a_terminal(cli: Cli) -> None:
    """The same document off a terminal (a script, a pipe) is the original compact line."""
    outcome = cli.run("schema", "--json")
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout.count("\n") == 1
    tty_outcome = cli.run("schema", "--json", tty_stdout=True)
    assert json.loads(outcome.stdout) == json.loads(tty_outcome.stdout)


def test_a_second_document_command_is_indented_on_a_terminal_too(cli: Cli) -> None:
    """The seam is generic: `schema probe`, a second document command, is indented as well."""
    outcome = cli.run("schema", "probe", "--json", tty_stdout=True)
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout.count("\n") > 1
    assert json.loads(outcome.stdout) == outcome.document


def test_ndjson_stream_records_stay_one_compact_line_regardless_of_tty() -> None:
    """O7a: `write_document` never indents unless its caller asks, so a stream record stays
    compact on a terminal too -- only the single-document path in `core.render` sets `indent`.
    """
    import io

    from provibench.core.output import write_document

    class _TtyStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _TtyStream()
    write_document(stream, {"seq": 1, "ok": True})
    write_document(stream, {"seq": 2, "ok": False})
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [{"seq": 1, "ok": True}, {"seq": 2, "ok": False}]
