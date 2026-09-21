"""I5 context classification independently of output-format defaults."""

from pathlib import Path

from promptreplay.core.context import Invocation, Process, Streams
from promptreplay.core.output import Format
from tests.conftest import FakeClock, FakeStream


def make_invocation(
    tmp_path: Path,
    *,
    stdout_tty: bool = True,
    env: dict[str, str] | None = None,
) -> Invocation:
    """Build an invocation with interactive stdin and stderr."""
    process = Process(
        streams=Streams(
            stdin=FakeStream(tty=True),
            stdout=FakeStream(tty=stdout_tty),
            stderr=FakeStream(tty=True),
        ),
        env=env or {},
        clock=FakeClock(),
        cwd=tmp_path,
        home=tmp_path,
    )
    return Invocation(program="promptreplay", process=process, declaration=())


def test_no_input_and_claimed_stdin_prevent_prompts(tmp_path: Path) -> None:
    no_input = make_invocation(tmp_path, env={"NO_INPUT": "1"})
    claimed = make_invocation(tmp_path)
    claimed.claim_stdin("the trace name")

    assert not no_input.can_prompt()
    assert not claimed.can_prompt()


def test_default_machine_format_keeps_prompts_but_blocks_a_session(tmp_path: Path) -> None:
    invocation = make_invocation(tmp_path)
    invocation.begin(Format.JSON)

    assert invocation.machine_readable
    assert invocation.interactive_context()
    assert invocation.can_prompt()
    assert not invocation.can_start_session()


def test_redirected_stdout_prevents_an_interactive_session(tmp_path: Path) -> None:
    invocation = make_invocation(tmp_path, stdout_tty=False)

    assert invocation.interactive_context()
    assert invocation.can_prompt()
    assert not invocation.can_start_session()
