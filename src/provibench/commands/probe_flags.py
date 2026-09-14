"""The flags `probe` and `sweep` share, and the request they build.

The two commands differ in one thing only — where their specs come from — so everything a
flag controls (the rungs, the repeats, the extras, the estimate, the confirmation) has to
mean the same in both, or the same endpoint measured by `probe` and by `sweep` would not be
the same measurement. The flags are declared once here and applied to both commands;
`ProbeRequest` is what crosses into `commands.probe.execute_probe`, which owns the flow.

`provibench.bench.*` and `commands.probe` pull in httpx and pydantic; their symbols are
imported only inside the functions that use them, so building the CLI stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import click

from provibench.core.errors import InvalidInput

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint
    from provibench.bench.probe import ProbeOptions
    from provibench.bench.probe_runs import SweepInfo
    from provibench.bench.targets import RunSpec
    from provibench.bench.trace import TraceEntry

_DEFAULT_REPEATS = "6,2,2"
_DEFAULT_GAP_S = 1.0
_DEFAULT_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """One probe-shaped run: the trace, the specs, and the flags that shape the protocol.

    `probe` fills `specs` from its arguments and `sweep` expands them from an endpoint list
    first; every other field comes from the flags both commands declare. `endpoints` and
    `sweep` belong to a sweep: the endpoint snapshot it already fetched (so the estimate is
    priced from the list the specs were built from, not a second fetch), and what
    `run.json` records about the expansion.
    """

    trace: str
    specs: Sequence[RunSpec]
    conversation: str | None = None
    rungs: str | None = None
    repeats: str = _DEFAULT_REPEATS
    gap: float = _DEFAULT_GAP_S
    warm: bool = False
    ttl: str | None = None
    throughput: bool = True
    strip_thinking: bool = False
    timeout: float = _DEFAULT_TIMEOUT_S
    budget: float | None = None
    yes: bool = False
    parallel: int = 1
    by_price: bool = False
    """Order the summaries — and the run directory — by measured effective price."""
    endpoints: Mapping[str, list[Endpoint]] | None = None
    sweep: SweepInfo | None = None


def options_for(request: ProbeRequest, conversation: Sequence[TraceEntry]) -> ProbeOptions:
    """The run's options, with rungs and repeats resolved against the conversation."""
    from pydantic import ValidationError

    from provibench.bench.probe import ProbeOptions
    from provibench.bench.rungs import parse_int_list

    try:
        options = ProbeOptions(
            rungs=None if request.rungs is None else parse_int_list(request.rungs),
            repeats=parse_int_list(request.repeats),
            gap_s=request.gap,
            warm=request.warm,
            throughput=request.throughput,
            ttl_s=None if request.ttl is None else parse_int_list(request.ttl),
            strip_thinking=request.strip_thinking,
            timeout_s=request.timeout,
        )
        return options.resolved(conversation)
    except (ValueError, ValidationError) as exc:
        # A bad list, a `--ttl` offset the `--gap` already passed, an impossible rung: all
        # of them are input errors, and all of them are caught before a request is sent.
        raise InvalidInput(_first_detail(exc)) from exc


def _first_detail(error: Exception) -> str:
    """One line for a validation error, so the message reads like the other input errors."""
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        return str(error.errors()[0]["msg"])
    return str(error)


def probe_options[FC: Callable[..., Any]](command: FC) -> FC:
    """Apply every `probe` flag to `command`, in the order `probe --help` lists them."""
    for option in reversed(
        (
            click.option(
                "--rungs",
                default=None,
                help="1-based turn indices to probe, comma-separated; each needs a "
                "following turn. Defaults to the smallest, middle and largest turn of "
                "the conversation",
            ),
            click.option(
                "--repeats",
                default=_DEFAULT_REPEATS,
                show_default=True,
                help="Warm reads per rung, comma-separated; the last value broadcasts to "
                "later rungs",
            ),
            click.option(
                "--gap",
                type=click.FloatRange(0.0),
                default=_DEFAULT_GAP_S,
                show_default=True,
                help="Seconds between warm reads",
            ),
            click.option(
                "--warm", is_flag=True, help="Send no nonce and measure the cache as found"
            ),
            click.option(
                "--ttl",
                default=None,
                help="Second offsets after the first served rung's warm reads to re-read "
                "its cache at, comma-separated and ascending (e.g. 60,300,900); off by "
                "default, it costs wall time",
            ),
            click.option(
                "--no-throughput",
                is_flag=True,
                help="Skip each rung's streamed generation request, so no TTFT, tok/s or "
                "fingerprint",
            ),
            click.option(
                "--conversation", default=None, help="Conversation key; defaults to the main one"
            ),
            click.option(
                "--strip-thinking",
                is_flag=True,
                help="Drop thinking blocks from assistant turns",
            ),
            click.option(
                "--timeout",
                type=click.FloatRange(0.1),
                default=_DEFAULT_TIMEOUT_S,
                show_default=True,
                help="Request timeout, seconds",
            ),
            click.option(
                "--budget",
                type=click.FloatRange(0.0),
                default=None,
                help="Refuse to run when the worst-case estimate exceeds this many USD",
            ),
            click.option(
                "--yes",
                is_flag=True,
                help="Skip the confirmation prompt; the budget check still applies",
            ),
        )
    ):
        command = option(command)
    return command
