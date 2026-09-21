"""The flags `probe` and `sweep` share, and the request they build.

The two commands differ in one thing only — where their endpoints come from — so everything
a flag controls (the rungs, the repeats, the extras, the estimate, the confirmation) has to
mean the same in both, or the same endpoint measured by `probe` and by `sweep` would not be
the same measurement. The flags are declared once here and applied to both commands;
`ProbeRequest` is what crosses into `commands.probe.execute_probe`, which owns the flow.

`promptreplay.bench.*` and `commands.probe` pull in httpx and pydantic; their symbols are
imported only inside the functions that use them, so building the CLI stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import click

from promptreplay.commands.dry_run import DRY_RUN_OPTION
from promptreplay.core.errors import InvalidInput
from promptreplay.core.params import TIMEOUT, parse_duration

if TYPE_CHECKING:
    from promptreplay.bench.openrouter import Endpoint
    from promptreplay.bench.precheck import CheckPlan
    from promptreplay.bench.probe import ProbeOptions
    from promptreplay.bench.selection import SweepInfo
    from promptreplay.bench.targets import RunSpec
    from promptreplay.bench.trace import TraceEntry

_DEFAULT_REPEATS = "6,2,2"
_DEFAULT_GAP_S = 1.0
_DEFAULT_TIMEOUT = "300s"
_DEFAULT_TIMEOUT_S = float(parse_duration(_DEFAULT_TIMEOUT))


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """One probe-shaped run: the trace, the endpoints, and the flags that shape the protocol.

    `probe` fills `specs` from its arguments and `sweep` expands them from an OpenRouter
    listing first; every other field comes from the flags both commands declare.
    `endpoints`, `pre_check` and `sweep` belong to a sweep: the OpenRouter listing it already
    fetched (so the estimate is priced from the same listing the run's endpoints were built
    from, not a second fetch), the availability check it planned (priced in the estimate,
    run after the confirmation), and what `run.json` records about the expansion.
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
    dry_run: bool = False
    parallel: int = 1
    by_price: bool = False
    """Order the summaries — and the run directory — by measured effective price."""
    endpoints: Mapping[str, list[Endpoint]] | None = None
    pre_check: CheckPlan | None = None
    """The availability check to run after the confirmation; `None` when the run does not."""
    sweep: SweepInfo | None = None


def options_for(request: ProbeRequest, conversation: Sequence[TraceEntry]) -> ProbeOptions:
    """The run's options, with rungs and repeats resolved against the conversation."""
    from pydantic import ValidationError

    from promptreplay.bench.probe import ProbeOptions
    from promptreplay.bench.rungs import parse_int_list

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
                type=TIMEOUT,
                default=_DEFAULT_TIMEOUT,
                show_default=True,
                metavar="DURATION",
                help="Request timeout: a duration such as 30s or 5m, or decimal seconds",
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
            DRY_RUN_OPTION,
        )
    ):
        command = option(command)
    return command
