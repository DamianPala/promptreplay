"""The phases between a probe request and its requests: estimate, confirmation, availability.

`commands.probe.execute_probe` is the order those phases run in; this module is what each of
them does, kept apart so that the flow reads as the list of steps it is. The order matters
and is the reason they sit together: the estimate is what the run costs (the availability
check included, priced from its plan), the budget is compared against that one number, the
confirmation has to come before anything is sent, and only then does the check visit the
candidates — a candidate it removes frees its `--top` slot for the next ranked one, and its
verdict lands in the run's `sweep` block.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
functions that use them, so building the CLI stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from provibench.core.confirm import require_confirmation
from provibench.core.context import Invocation

if TYPE_CHECKING:
    from provibench.bench.estimate import PreCheckCost, SpecEstimate, SpecPrices, UpperBound
    from provibench.bench.openrouter import Endpoint
    from provibench.bench.precheck import CheckPlan, PreCheckResult
    from provibench.bench.probe import ProbeOptions
    from provibench.bench.probe_models import ProbeResult
    from provibench.bench.selection import SelectionDrop, SweepInfo
    from provibench.bench.targets import RunSpec
    from provibench.bench.trace import TraceEntry
    from provibench.commands.probe_flags import ProbeRequest


@dataclass(frozen=True, slots=True)
class Checked:
    """What the availability check left: the specs to probe, and the candidates it removed."""

    specs: list[RunSpec]
    drops: list[SelectionDrop]
    precheck: list[ProbeResult] = field(default_factory=list["ProbeResult"])
    """Every request the availability check sent, role `precheck`; persisted separately from
    any spec's own records (see `bench.probe_runs.write_probe_run`)."""


def planned_specs(request: ProbeRequest) -> list[RunSpec]:
    """The specs the estimate prices and the confirmation names.

    With `--top N` that is the N best candidates, which are the specs the run probes unless
    the availability check removes one and promotes the next ranked candidate (priced the
    same way); without it, every spec. The references are always included.
    """
    plan = request.pre_check
    if plan is None or plan.keep is None:
        return list(request.specs)
    candidates = {spec.label for spec in plan.candidates}
    return [
        *plan.candidates[: plan.keep],
        *(spec for spec in request.specs if spec.label not in candidates),
    ]


def estimates(
    request: ProbeRequest,
    selected: Sequence[TraceEntry],
    options: ProbeOptions,
    prices: Mapping[str, SpecPrices],
) -> list[SpecEstimate]:
    """One worst case per spec the run means to probe; the check is priced once, by the plan."""
    from provibench.bench.estimate import probe_estimate

    return [
        probe_estimate(spec, selected, options, prices.get(spec.label))
        for spec in planned_specs(request)
    ]


def upper_bound_estimate(
    request: ProbeRequest,
    selected: Sequence[TraceEntry],
    options: ProbeOptions,
    prices: Mapping[str, SpecPrices],
    *,
    pre_check: PreCheckCost | None,
) -> UpperBound | None:
    """What a `--top` run costs at most, or `None` when there is no cut to bound.

    The estimate prices the N best-ranked candidates, but the availability check decides
    which candidates the run probes: any N of them can be the ones that survive it, so the
    priced rows are the expectation and this is the ceiling. A run without `--top` already
    prices every spec it will probe, and a candidate with no listed price cannot be ranked
    by one — both say `None`, which leaves the total in charge of the budget.
    """
    from provibench.bench.estimate import UpperBound, estimate_total, probe_estimate

    plan = request.pre_check
    if plan is None or plan.keep is None:
        return None
    keep = plan.keep
    if any(prices.get(spec.label) is None for spec in plan.candidates):
        return None

    def listed(spec: RunSpec) -> float:
        price = prices.get(spec.label)
        return 0.0 if price is None else price.prices.input

    candidates = {spec.label for spec in plan.candidates}
    specs = [
        *sorted(plan.candidates, key=listed, reverse=True)[:keep],
        *(spec for spec in request.specs if spec.label not in candidates),
    ]
    total = estimate_total(
        [probe_estimate(spec, selected, options, prices.get(spec.label)) for spec in specs],
        pre_check=pre_check,
    )
    return None if total is None else UpperBound(keep=keep, usd=total)


def confirm(
    invocation: Invocation,
    estimates: Sequence[SpecEstimate],
    options: ProbeOptions,
    specs: Sequence[RunSpec],
    *,
    yes: bool,
    pre_check: PreCheckCost | None = None,
    upper_bound: UpperBound | None = None,
) -> None:
    """Ask before anything is sent, or return for `--yes`.

    The question is the estimate in one line: how many requests, to what, over which turns,
    for how much at worst. A planned availability check is named as part of it, because it is
    priced in that number and runs as soon as the caller agrees. So is an upper bound, since
    that one is the number the caller can really be billed for: `--top` prices the N best
    candidates and the check decides which of them the run probes.
    """
    from provibench.bench.estimate import estimate_total, same_amount

    total = estimate_total(estimates, pre_check=pre_check)
    worst = "an unknown amount" if total is None else f"up to ${total:.4f} at worst, no cache hit"
    requests = len(specs) * _requests(options)
    question = (
        f"Send {requests} probe request(s) to {len(specs)} spec(s) over turn(s) "
        f"{options.rungs}, spending {worst}"
    )
    if pre_check is not None:
        question += f"; the availability check runs first, up to {pre_check.requests} candidate(s)"
    if upper_bound is not None and not same_amount(upper_bound.usd, total):
        question += (
            f"; if the check promotes the {upper_bound.keep} priciest candidates, "
            f"up to ${upper_bound.usd:.4f}"
        )
    if options.ttl_s:
        question += f"; TTL re-reads at {options.ttl_s}s on the first rung"
    unpriced = [estimate.label for estimate in estimates if estimate.usd is None]
    if unpriced:
        question += f"; no listed price for {', '.join(unpriced)}"
    require_confirmation(invocation, question=question, yes=yes)


def resolve_index(request: ProbeRequest) -> tuple[dict[str, list[Endpoint]], list[str]]:
    """The endpoint index a sweep already fetched, or a fresh lookup for a plain probe."""
    from provibench.commands.run_specs import endpoint_index

    if request.endpoints is None:
        return endpoint_index(request.specs)
    return dict(request.endpoints), []


def checked_specs(
    request: ProbeRequest,
    selected: Sequence[TraceEntry],
    options: ProbeOptions,
    invocation: Invocation,
) -> Checked:
    """The availability phase, or every spec when the run planned no check."""
    if request.pre_check is None:
        return Checked(list(request.specs), [])
    return run_check(invocation, request.pre_check, request.specs, selected, options)


def run_check(
    invocation: Invocation,
    plan: CheckPlan,
    specs: Sequence[RunSpec],
    entries: Sequence[TraceEntry],
    options: ProbeOptions,
) -> Checked:
    """Visit the planned candidates and report what the run will probe instead.

    Every candidate is checked in ranking order until `--top` of them have survived, so a
    removal frees its slot for the next ranked candidate instead of filling it. The native
    references are neither checked nor cut: a tag is not how they were chosen.
    """
    from provibench.bench.precheck import precheck_candidates

    results = asyncio.run(
        precheck_candidates(
            plan.candidates,
            entries,
            options,
            invocation.env,
            keep=plan.keep,
            on_result=lambda result: progress_line(invocation, result),
        )
    )
    candidates = {spec.label for spec in plan.candidates}
    return Checked(
        specs=[
            *(result.spec for result in results if result.kept),
            *(spec for spec in specs if spec.label not in candidates),
        ],
        drops=[_drop(result) for result in results if not result.kept],
        precheck=[result.record for result in results if result.record is not None],
    )


def progress_line(invocation: Invocation, result: PreCheckResult) -> None:
    """One line per checked candidate: what came back, and, on a failure, why it is out."""
    line = f"pre-check {result.spec.label} {result.status} {result.latency_ms:.0f}ms"
    if result.reason is not None:
        line += f" {result.reason}"
    invocation.message(line)


def recorded_sweep(sweep: SweepInfo | None, checked: Checked) -> SweepInfo | None:
    """The run's sweep block, with the candidates the availability check removed added.

    The verdicts are only known once the check has run, so the block the sweep built is
    copied rather than mutated: the caller's record stays what the criteria chose.
    """
    if sweep is None or not checked.drops:
        return sweep
    return sweep.model_copy(update={"dropped": [*sweep.dropped, *checked.drops]})


def _requests(options: ProbeOptions) -> int:
    """How many requests one spec sends: cold and warm per rung, plus the extras."""
    per_spec = sum(1 + count for count in options.repeats)
    if options.throughput:
        per_spec += len(options.repeats)
    if options.ttl_s:
        per_spec += len(options.ttl_s)
    return per_spec


def _drop(result: PreCheckResult) -> SelectionDrop:
    from provibench.bench.selection import SelectionDrop

    return SelectionDrop.for_spec(result.spec, result.reason or "unavailable", checked=True)
