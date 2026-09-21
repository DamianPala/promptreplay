"""Shared fixtures: an in-process CLI runner with injected boundaries, and a subprocess runner."""

import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from promptreplay.app import main
from promptreplay.core.context import Process, Streams
from promptreplay.core.documents import Document, as_document

FIXTURE_TABLE = Path(__file__).parent / "fixtures" / "litellm-prices.json"
"""A trimmed real capture of LiteLLM's `model_prices_and_context_window.json`."""


@pytest.fixture(autouse=True)
def no_price_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the LiteLLM table over the network; the fetch boundary fails instead.

    The failure is the offline case the cache policy is built for, so a test that never
    installs a copy exercises the `n/a` path. A test that wants real prices calls
    `install_price_cache` (the fixture table as the cached copy) or replaces
    `promptreplay.bench.prices.fetch_payload` with its own fetcher.
    """

    async def offline(client: httpx.AsyncClient) -> dict[str, object]:
        raise httpx.ConnectError("no network in tests", request=client.build_request("GET", "x"))

    monkeypatch.setattr("promptreplay.bench.prices.fetch_payload", offline)


@pytest.fixture(autouse=True)
def no_reported_average_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches OpenRouter's reported-average feed; the fetch boundary returns `None`.

    That is the same outcome a real fetch failure produces, so a test that never replaces
    `promptreplay.bench.openrouter_stats.fetch_reported_average` exercises the "unavailable"
    path. A test that wants a figure replaces that name with its own fetcher.
    """

    async def offline(client: httpx.AsyncClient, model: str, *, day: str) -> None:
        del client, model, day
        return None

    monkeypatch.setattr("promptreplay.bench.openrouter_stats.fetch_reported_average", offline)


class FakeClock:
    """A clock that advances only when something sleeps, so waits finish instantly."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.time = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.time += seconds


class FakeStream(io.StringIO):
    """A text stream whose TTY state the test chooses."""

    def __init__(self, initial: str = "", *, tty: bool = False) -> None:
        super().__init__(initial)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@dataclass(frozen=True)
class Outcome:
    """What one invocation produced."""

    code: int
    stdout: str
    stderr: str

    @property
    def document(self) -> Document:
        """stdout parsed as one JSON document."""
        document = as_document(json.loads(self.stdout))
        assert document is not None, self.stdout
        return document

    @property
    def records(self) -> list[Document]:
        """stdout parsed as NDJSON."""
        records = [as_document(json.loads(line)) for line in self.stdout.splitlines() if line]
        return [record for record in records if record is not None]

    @property
    def error(self) -> Document:
        """The F3 `error` object from the last non-empty stderr line."""
        lines = [line for line in self.stderr.splitlines() if line.strip()]
        assert lines, "stderr is empty"
        envelope = as_document(json.loads(lines[-1]))
        assert envelope is not None and set(envelope) == {"error"}, lines[-1]
        error = as_document(envelope["error"])
        assert error is not None
        return error


class Cli:
    """Runs `promptreplay` in-process with a temporary home and working directory."""

    def __init__(self, root: Path, clock: FakeClock) -> None:
        self.root = root
        self.clock = clock
        self.home = root / "home"
        self.home.mkdir()
        self.env: dict[str, str] = {"HOME": str(self.home)}

    def run(
        self,
        *args: str,
        stdin: str = "",
        tty: bool = False,
        tty_stdin: bool | None = None,
        tty_stdout: bool | None = None,
        tty_stderr: bool | None = None,
        env: Mapping[str, str] | None = None,
        unset: Sequence[str] = (),
    ) -> Outcome:
        environment = {**self.env, **(env or {})}
        for name in unset:
            environment.pop(name, None)
        stdout = FakeStream(tty=tty if tty_stdout is None else tty_stdout)
        stderr = FakeStream(tty=tty if tty_stderr is None else tty_stderr)
        streams = Streams(
            stdin=FakeStream(stdin, tty=tty if tty_stdin is None else tty_stdin),
            stdout=stdout,
            stderr=stderr,
        )
        process = Process(
            streams=streams,
            env=environment,
            clock=self.clock,
            cwd=self.root,
            home=self.home,
        )
        code = main(list(args), process=process)
        return Outcome(code, stdout.getvalue(), stderr.getvalue())


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cli(tmp_path: Path, clock: FakeClock) -> Cli:
    return Cli(tmp_path, clock)


def install_price_cache(cli: Cli, *, age_s: float = 0.0) -> Path:
    """The fixture table as this invocation's cached LiteLLM copy, `age_s` seconds old.

    Age is an mtime, which is what the freshness rule reads: a copy stamped a week and a
    day back is refetched even though it was written a moment ago.
    """
    path = cli.home / ".cache" / "promptreplay" / "litellm-prices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIXTURE_TABLE.read_text(encoding="utf-8"), encoding="utf-8")
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


@dataclass(frozen=True)
class BenchPaths:
    """Fresh traces/runs directories and a targets file, wired up via env vars.

    `record`/`inspect`/`replay`/`report` all resolve their storage through the
    `traces_dir`/`runs_dir`/`targets_path` settings; pointing those at a scratch area
    via `PROMPTREPLAY_*` env vars (rather than relying on `Cli`'s cwd/home defaults) keeps
    bench-command tests independent of the generic config precedence tests.
    """

    traces_dir: Path
    runs_dir: Path
    targets_path: Path
    env: dict[str, str]


@pytest.fixture
def bench_paths(tmp_path: Path) -> BenchPaths:
    traces_dir = tmp_path / "bench" / "traces"
    runs_dir = tmp_path / "bench" / "runs"
    targets_path = tmp_path / "bench" / "targets.toml"
    traces_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    env = {
        "PROMPTREPLAY_TRACES_DIR": str(traces_dir),
        "PROMPTREPLAY_RUNS_DIR": str(runs_dir),
        "PROMPTREPLAY_TARGETS": str(targets_path),
    }
    return BenchPaths(traces_dir, runs_dir, targets_path, env)


def run_process(
    args: Sequence[str],
    *,
    env: Mapping[str, str],
    stdin: str | None = None,
    timeout: float = 30.0,
    hold_stdin: bool = False,
) -> Outcome:
    """Run the installed tool as a subprocess with only the given environment.

    `hold_stdin` keeps stdin an open pipe that is never written and never closed
    while the child runs. A tool that reads stdin blocks until `timeout`; a tool
    that does not returns as usual. Its output is drained only after the child
    exits, so use it for commands that stay well under the pipe buffer.
    """
    command = [sys.executable, "-m", "promptreplay", *args]
    environment = {"PATH": os.environ["PATH"], **env}
    if hold_stdin:
        return _run_holding_stdin(command, env=environment, timeout=timeout)
    completed = subprocess.run(
        command,
        input=stdin,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout,
        check=False,
    )
    return Outcome(completed.returncode, completed.stdout, completed.stderr)


def _run_holding_stdin(
    command: Sequence[str], *, env: Mapping[str, str], timeout: float
) -> Outcome:
    """Popen with an open stdin pipe; collect output only after the child exits."""
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    ) as process:
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        stdout, stderr = process.communicate()
        return Outcome(code, stdout, stderr)
