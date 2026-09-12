"""O3d: terminal control characters in human output are visible text, never active controls."""

from pathlib import Path

from provibench.core.color import ColorMode
from provibench.core.completion import render_completion
from provibench.core.config_command import render_settings
from provibench.core.confirm import require_confirmation
from provibench.core.context import Invocation, Process, Streams
from provibench.core.documents import as_document
from provibench.core.output import write_plain_line
from provibench.core.render import render_document
from provibench.core.terminal_text import escape_terminal_text
from tests.conftest import Cli, FakeClock, FakeStream

OSC = "\x1b]52;c;clipboard\x07"
CSI = "\x1b[31m"


def make_invocation(
    *, color: ColorMode = ColorMode.NEVER, stdin: str = ""
) -> tuple[Invocation, FakeStream, FakeStream]:
    """Build an invocation with terminal streams for direct human-renderer tests."""
    stdout = FakeStream(tty=True)
    stderr = FakeStream(tty=True)
    invocation = Invocation(
        program="provibench",
        process=Process(
            streams=Streams(stdin=FakeStream(stdin, tty=True), stdout=stdout, stderr=stderr),
            env={"TERM": "xterm"},
            clock=FakeClock(),
            cwd=Path("."),
            home=Path("."),
        ),
        declaration=(),
    )
    invocation.flags.color = color
    return invocation, stdout, stderr


def test_escape_terminal_text_makes_c0_c1_and_del_visible() -> None:
    assert escape_terminal_text("line\nring\x07escape\x1bCSI\x9bdel\x7f") == (
        "line\\x0aring\\x07escape\\x1bCSI\\x9bdel\\x7f"
    )


def test_message_escapes_controls_with_and_without_color() -> None:
    for color in (ColorMode.NEVER, ColorMode.ALWAYS):
        invocation, _stdout, stderr = make_invocation(color=color)
        invocation.message(f"problem {OSC} {CSI}")
        rendered = stderr.getvalue()
        assert OSC not in rendered and CSI not in rendered
        assert "\\x1b]52;c;clipboard\\x07" in rendered and "\\x1b[31m" in rendered


def test_confirmation_prompt_escapes_controls_even_when_quiet() -> None:
    invocation, _stdout, stderr = make_invocation(stdin="yes\n")
    invocation.flags.quiet = True

    require_confirmation(invocation, question=f"Approve {OSC} {CSI}", yes=False)

    rendered = stderr.getvalue()
    assert OSC not in rendered and CSI not in rendered
    assert "\\x1b]52;c;clipboard\\x07" in rendered and "\\x1b[31m" in rendered
    assert "[y/N]" in rendered


def test_generic_and_plain_renderers_escape_values_keys_next_and_cursor() -> None:
    invocation, stdout, _stderr = make_invocation()
    render_document(
        invocation,
        {
            f"field{CSI}": f"value{OSC}",
            "next": ["provibench", "replay", f"trace{CSI}"],
        },
    )
    render_document(
        invocation,
        {
            "items": [{f"column{CSI}": f"cell{OSC}"}],
            "has_more": True,
            "next_cursor": f"cursor{CSI}",
        },
    )
    write_plain_line(stdout, f"plain{OSC}")
    rendered = stdout.getvalue()
    assert OSC not in rendered and CSI not in rendered
    assert "Field\\x1b[31m" in rendered and "value\\x1b]52;c;clipboard\\x07" in rendered
    assert "trace\\x1b[31m" in rendered and "cursor\\x1b[31m" in rendered
    assert "plain\\x1b]52;c;clipboard\\x07" in rendered


def test_config_diagnostics_and_completion_location_escape_without_changing_json(cli: Cli) -> None:
    config = cli.root / f"config{OSC}.toml"
    config.write_text("")
    human = cli.run("config", "show", "--config", str(config), "--color", "never", tty=True)
    machine = cli.run("config", "show", "--config", str(config), "--json")
    assert OSC not in human.stdout
    settings = as_document(machine.document["settings"]) or {}
    config_setting = as_document(settings["config_path"]) or {}
    assert config_setting["value"] == str(config)

    invocation, stdout, _stderr = make_invocation()
    render_settings(
        invocation,
        {"settings": {"config_path": {"value": f"path{OSC}", "source": "flag"}}},
    )
    assert OSC not in stdout.getvalue() and "path\\x1b]52;c;clipboard\\x07" in stdout.getvalue()

    invocation, stdout, _stderr = make_invocation()
    render_completion(invocation, {"shell": "bash", "path": f"path{OSC}", "changed": True})
    assert OSC not in stdout.getvalue() and "path\\x1b]52;c;clipboard\\x07" in stdout.getvalue()
