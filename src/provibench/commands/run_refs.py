"""Resolving a RUN argument: a run directory, a trace's newest run, or `latest`/`previous`.

Both `report` and `compare` name runs the same way, so the resolution lives here once: a
step back through time is a comparison between two runs of one trace, and the two commands
must not disagree about which directory that is.

A run's order is its directory name — the `%Y%m%d-%H%M%S` stamp the run was written
under — because that is what a reader sees in `ls` and what every command has ordered by
since runs were persisted.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from provibench.core.errors import NotFound

__all__ = ["resolve_run_dir"]

LATEST = "latest"
PREVIOUS = "previous"
"""The two run names that are not names of runs: the newest of a trace, and the one before."""


def resolve_run_dir(
    run: str,
    runs_dir: Path,
    cwd: Path,
    *,
    trace: str | None = None,
    previous: bool = False,
) -> Path:
    """The run directory RUN names, or `not_found` with what to pass instead.

    `run` resolves, in this order, as: a path to a run directory (absolute, or relative to
    the working directory), then a trace name (its newest run), then `latest` and — where
    the caller opts in with `previous` — `previous`, the two newest runs of the trace
    `--trace` names. Without `--trace` the default trace is the one the newest run belongs
    to, so `latest` and `previous` always name two runs of the same trace, which is the
    only pair `compare` accepts.
    """
    candidate = Path(run)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if (candidate / "run.json").is_file():
        return candidate
    if run == LATEST or (previous and run == PREVIOUS):
        return _recent(run, runs_dir, trace)
    trace_dir = runs_dir / run
    newest = newest_run_dir(trace_dir.glob("*")) if trace_dir.is_dir() else None
    if newest is None:
        names = f"{LATEST!r}/{PREVIOUS!r}" if previous else repr(LATEST)
        raise NotFound(
            f"No run found for {run!r}",
            hint=f"Pass a run directory, a trace name under {runs_dir}, or {names}",
        )
    return newest


def newest_run_dir(candidates: Iterable[Path]) -> Path | None:
    """The newest run directory among these, or `None` when none of them is one."""
    runs = run_dirs(candidates)
    return runs[-1] if runs else None


def run_dirs(candidates: Iterable[Path]) -> list[Path]:
    """The paths that are run directories, ordered by their stamp."""
    return sorted(
        (path for path in candidates if path.is_dir() and (path / "run.json").is_file()),
        key=lambda path: path.name,
    )


def _recent(run: str, runs_dir: Path, trace: str | None) -> Path:
    """`latest` or `previous`: two adjacent runs of one trace, newest last."""
    if trace is None:
        newest = newest_run_dir(runs_dir.glob("*/*"))
        if newest is None:
            raise NotFound(
                f"No runs under {runs_dir}",
                hint="Run something first: provibench probe, or provibench sweep",
            )
        trace = newest.parent.name
    found = run_dirs((runs_dir / trace).glob("*"))
    if not found:
        known = ", ".join(_traces(runs_dir)) or "(none)"
        raise NotFound(
            f"No run found for trace {trace!r}", hint=f"Traces under {runs_dir}: {known}"
        )
    if run == LATEST:
        return found[-1]
    if len(found) < 2:
        raise NotFound(
            f"Trace {trace!r} has one run, so there is no run before the newest",
            hint="Compare two run directories, or record a second run of that trace",
        )
    return found[-2]


def _traces(runs_dir: Path) -> list[str]:
    """The trace names under the runs dir, without repeats."""
    return sorted({path.parent.name for path in run_dirs(runs_dir.glob("*/*"))})
