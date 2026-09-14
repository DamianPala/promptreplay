"""The command-layer wiring `probe` and `replay` share: targets, run specs, conversations.

`provibench.bench.targets` and `provibench.bench.trace` pull in pydantic; their symbols are
imported only inside the functions that use them, so building the CLI stays cheap.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from provibench.core.context import Invocation
from provibench.core.errors import InvalidInput, NotFound

if TYPE_CHECKING:
    from provibench.bench.estimate import SpecEstimate, SpecPrices
    from provibench.bench.openrouter import Endpoint
    from provibench.bench.targets import RunSpec, Target
    from provibench.bench.trace import TraceEntry


def load_targets(invocation: Invocation) -> dict[str, Target]:
    """The selected targets file, or the packaged defaults when none is configured."""
    targets_path = Path(invocation.setting("targets_path") or "")
    if targets_path.is_file():
        return _parse_targets(targets_path)
    invocation.message(f"No targets file at {targets_path}; using the packaged defaults")
    packaged = resources.files("provibench.data").joinpath("targets.toml")
    with resources.as_file(packaged) as path:
        return _parse_targets(path)


def parse_specs(raw_specs: Iterable[str], targets: dict[str, Target]) -> list[RunSpec]:
    """One `<target>:<model>[@provider,...]` run spec per argument, or `invalid_input`."""
    from provibench.bench.targets import parse_run_spec

    specs: list[RunSpec] = []
    for raw in raw_specs:
        try:
            specs.append(parse_run_spec(raw, targets))
        except ValueError as exc:
            raise InvalidInput(str(exc)) from exc
    return specs


def endpoint_index(run_specs: Sequence[RunSpec]) -> tuple[dict[str, list[Endpoint]], list[str]]:
    """The OpenRouter endpoint snapshot the estimates and the run record use, best-effort.

    `httpx` and the endpoint client are reached through `provibench.bench.estimate`, which
    imports them inside the call, so a test can replace them at their own module.
    """
    import asyncio

    from provibench.bench.estimate import fetch_endpoint_index

    models = [spec.model for spec in run_specs if spec.target.kind == "openrouter"]
    if not models:
        return {}, []
    return asyncio.run(fetch_endpoint_index(models))


def spec_price_map(
    run_specs: Sequence[RunSpec], index: dict[str, list[Endpoint]]
) -> dict[str, SpecPrices]:
    """The listed price of every spec that has one, keyed by label."""
    from provibench.bench.estimate import spec_prices

    prices: dict[str, SpecPrices] = {}
    for spec in run_specs:
        listed = spec_prices(spec, index)
        if listed is not None:
            prices[spec.label] = listed
    return prices


def unpriced_labels(estimates: Sequence[SpecEstimate]) -> list[str]:
    """The specs whose worst case could not be computed, in the order given."""
    return [estimate.label for estimate in estimates if estimate.usd is None]


def check_budget(estimates: Sequence[SpecEstimate], budget: float | None, *, hint: str) -> None:
    """Refuse a budget the estimate cannot cover, before anything is sent.

    An unpriced spec makes the total unknowable, so `--budget` is refused outright rather
    than checked against the priced part of it; the hint is the caller's own advice for a
    total that is merely too big.
    """
    from provibench.bench.estimate import estimate_total

    if budget is None:
        return
    unpriced = unpriced_labels(estimates)
    if unpriced:
        raise InvalidInput(
            f"--budget cannot be checked: no listed price for {', '.join(unpriced)}",
            hint="Add a prices table for that model to targets.toml, or drop --budget",
        )
    total = estimate_total(estimates)
    if total is not None and total > budget:
        raise InvalidInput(
            f"The worst-case estimate ${total:.4f} exceeds --budget ${budget:.4f}",
            hint=hint,
        )


def select_conversation(
    entries: list[TraceEntry], conversation: str | None
) -> tuple[list[TraceEntry], str]:
    """The selected conversation's turns in order, with the key it is stored under."""
    from provibench.bench.trace import group_conversations, main_conversation

    groups = group_conversations(entries)
    if conversation is not None:
        if conversation not in groups:
            raise NotFound(f"No conversation {conversation!r} in this trace")
        return sorted(groups[conversation], key=lambda e: e.seq), conversation
    main = main_conversation(entries)
    if not main:
        raise InvalidInput("This trace has no conversations to replay")
    return sorted(main, key=lambda e: e.seq), main[0].conversation


def _parse_targets(path: Path) -> dict[str, Target]:
    from provibench.bench.targets import load_targets

    try:
        return load_targets(path)
    except ValueError as exc:
        raise InvalidInput(str(exc)) from exc
