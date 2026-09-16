"""The command-layer wiring `probe` and `replay` share: targets, run specs, conversations.

`provibench.bench.targets` and `provibench.bench.trace` pull in pydantic; their symbols are
imported only inside the functions that use them, so building the CLI stays cheap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from provibench.core.context import Invocation
from provibench.core.errors import InvalidInput, NotFound, OperationFailed

if TYPE_CHECKING:
    from provibench.bench.estimate import PreCheckCost, SpecEstimate, SpecPrices, UpperBound
    from provibench.bench.openrouter import Endpoint
    from provibench.bench.prices import PriceTable
    from provibench.bench.targets import RunSpec, Target
    from provibench.bench.trace import TraceEntry

_ZDR_TIMEOUT_S = 30.0


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
    """The OpenRouter endpoint snapshot the estimates and the run record use, best-effort."""
    models = [spec.model for spec in run_specs if spec.target.kind == "openrouter"]
    if not models:
        return {}, []
    return fetch_model_endpoints(models)


def fetch_model_endpoints(
    models: Sequence[str], *, api_key: str | None = None
) -> tuple[dict[str, list[Endpoint]], list[str]]:
    """The OpenRouter endpoint lists of these models, plus one note per failed lookup.

    `api_key` makes the list carry the health percentiles, which OpenRouter returns only for
    a keyed request; a sweep passes it whenever its target has one, because the listing's p50
    columns and the default price sort's throughput tie-break read them too.
    """
    import asyncio

    from provibench.bench.estimate import fetch_endpoint_index

    return asyncio.run(fetch_endpoint_index(models, api_key=api_key))


def fetch_zdr_tags(model: str) -> set[str]:
    """The Zero Data Retention endpoint tags of one model, or `operation_failed`.

    The ZDR list covers every model at once and needs no API key; it is fetched per sweep
    because a sweep is where the answer is used, and one request is cheaper to keep simple
    than a cache of a list that changes with other people's provider settings.
    """
    import asyncio

    import httpx

    from provibench.bench.openrouter import fetch_zdr_endpoints

    async def fetch() -> set[str]:
        async with httpx.AsyncClient(timeout=_ZDR_TIMEOUT_S) as client:
            endpoints = await fetch_zdr_endpoints(client)
        return {endpoint.tag for endpoint in endpoints if endpoint.model_id == model}

    try:
        return asyncio.run(fetch())
    except (httpx.HTTPError, ValueError) as exc:
        raise OperationFailed(f"The Zero Data Retention list could not be fetched: {exc}") from exc


def gateway_target(name: str | None, targets: Mapping[str, Target]) -> Target:
    """The `kind = "openrouter"` target whose endpoints a sweep pins its specs to."""
    if name is None:
        first = next((t for t in targets.values() if t.kind == "openrouter"), None)
        if first is None:
            known = ", ".join(sorted(targets)) or "(none)"
            raise InvalidInput(
                'No OpenRouter target to sweep: targets.toml has no kind="openrouter" entry',
                hint=f"Add one, or pass a run spec to probe directly; known targets: {known}",
            )
        return first
    target = targets.get(name)
    if target is None:
        known = ", ".join(sorted(targets)) or "(none)"
        raise InvalidInput(
            f"Unknown target {name!r} for --target; known targets: {known}",
        )
    if target.kind != "openrouter":
        raise InvalidInput(
            f"--target {name!r} is kind={target.kind!r}; a sweep lists the endpoints of a "
            'kind="openrouter" target',
        )
    return target


def filtered_endpoints(
    model: str,
    endpoints: Sequence[Endpoint],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[Endpoint]:
    """The endpoints the tag filters select, in the order the endpoint list gave them.

    The filters are prefixes of the tag (`--include novita` keeps `novita/fp8`), and a tag no
    endpoint has is an input error rather than a silently thinner run.
    """
    kept = _kept_tags([endpoint.tag for endpoint in endpoints], include=include, exclude=exclude)
    if not kept:
        untagged = untagged_count(endpoints)
        lost = f"; {untagged} of {len(endpoints)} came back without a tag" if untagged else ""
        raise InvalidInput(
            f"The endpoint list of {model!r} is empty after filtering{lost}",
            hint="Drop a --exclude, or list the endpoints with: provibench endpoints MODEL",
        )
    first: dict[str, Endpoint] = {}
    for endpoint in endpoints:
        first.setdefault(endpoint.tag, endpoint)
    return [first[tag] for tag in kept]


def native_specs(model: str, targets: Mapping[str, Target]) -> list[RunSpec]:
    """One spec per `kind = "anthropic"` target whose aliases map `model` to its own name.

    A native target is selected by the model, not by a tag: it carries the slug through
    `[targets.<name>.aliases]`, which is an explicit statement of which of its models the
    slug means, and its row is the reference the measured endpoints are read against.
    """
    from provibench.bench.targets import RunSpec

    return [
        RunSpec(target=target, model=target.aliases[model])
        for target in targets.values()
        if target.kind == "anthropic" and model in target.aliases
    ]


def untagged_count(endpoints: Sequence[Endpoint]) -> int:
    """How many endpoints the gateway returned without a tag: they cannot be pinned.

    An endpoint without one has no `@tag` to name it by, so a sweep cannot probe it at all;
    it is dropped, and the count is what lets a caller say so rather than let the row go
    missing or an empty run blame the filters.
    """
    return sum(1 for endpoint in endpoints if not endpoint.tag)


def _kept_tags(tags: Sequence[str], *, include: Sequence[str], exclude: Sequence[str]) -> list[str]:
    """The endpoint tags the filters select, in the order the endpoint list gave them."""
    wanted = _selected(tags, include, flag="--include") if include else set(tags)
    unwanted: set[str] = _selected(tags, exclude, flag="--exclude") if exclude else set()
    return [tag for tag in dict.fromkeys(tags) if tag and tag in wanted and tag not in unwanted]


def _selected(tags: Sequence[str], patterns: Sequence[str], *, flag: str) -> set[str]:
    """Every tag one filter selects, or an input error naming the tags that do exist."""
    selected: set[str] = set()
    for pattern in patterns:
        matches = {tag for tag in tags if tag.startswith(pattern)}
        if not matches:
            available = ", ".join(sorted(set(tags))) or "(none)"
            raise InvalidInput(
                f"{flag} {pattern!r} matches no endpoint; available tags: {available}",
                hint="Tags are the endpoint slugs that `provibench endpoints MODEL` prints",
            )
        selected |= matches
    return selected


def validate_pinned_providers(
    run_specs: Sequence[RunSpec], index: Mapping[str, list[Endpoint]]
) -> None:
    """Fail fast when a pinned gateway spec names a tag or provider no endpoint carries.

    `spec_prices` already returns `None` for a pin that matches nothing, which reads to
    `--budget` as merely unpriced and invites pricing a provider that will never answer
    (feedback: the hint suggested adding a targets.toml price for a phantom endpoint). The
    tags come from the same endpoint list `sweep --include` validates a prefix against, so
    both name the unknown-tag error the same way and point at the same command.

    A spec whose model has no entry in `index` is left alone: the lookup itself failed or
    was never attempted, which is reported separately, and guessing "no such provider" from
    an empty list would be wrong as often as right.
    """
    from provibench.bench.openrouter import normalize_provider

    for spec in run_specs:
        if spec.target.kind != "openrouter" or not spec.providers:
            continue
        endpoints = index.get(spec.model, [])
        if not endpoints:
            continue
        wanted = {normalize_provider(name) for name in spec.providers}
        matched = any(
            normalize_provider(endpoint.tag) in wanted
            or normalize_provider(endpoint.provider_name) in wanted
            for endpoint in endpoints
        )
        if matched:
            continue
        tags = ", ".join(sorted({endpoint.tag for endpoint in endpoints if endpoint.tag}))
        bad = ", ".join(repr(name) for name in spec.providers)
        raise InvalidInput(
            f"no endpoint tagged {bad} serves {spec.model}; tags: {tags or '(none)'}",
            hint="provibench endpoints MODEL",
        )


def spec_price_map(
    run_specs: Sequence[RunSpec],
    index: Mapping[str, list[Endpoint]],
    *,
    table: PriceTable | None = None,
) -> dict[str, SpecPrices]:
    """The listed price of every spec that has one, keyed by label; `table` prices natives."""
    from provibench.bench.estimate import spec_prices

    prices: dict[str, SpecPrices] = {}
    for spec in run_specs:
        listed = spec_prices(spec, index, table)
        if listed is not None:
            prices[spec.label] = listed
    return prices


def unpriced_labels(estimates: Sequence[SpecEstimate]) -> list[str]:
    """The specs whose worst case could not be computed, in the order given."""
    return [estimate.label for estimate in estimates if estimate.usd is None]


def check_budget(
    estimates: Sequence[SpecEstimate],
    budget: float | None,
    *,
    hint: str,
    pre_check: PreCheckCost | None = None,
    upper_bound: UpperBound | None = None,
) -> None:
    """Refuse a budget the estimate cannot cover, before anything is sent.

    An unpriced spec makes the total unknowable, so `--budget` is refused outright rather
    than checked against the priced part of it; the hint is the caller's own advice for a
    total that is merely too big. The availability check is part of that total — it is the
    run's own spend, priced from the plan — so a refusal says how much of it the check is
    when dropping the check would fit. A `--top` run brings an `upper_bound` too: the check
    can promote the priciest candidates under it, and that is the number the budget has to
    cover, so the refusal names it.
    """
    from provibench.bench.estimate import estimate_total, same_amount

    if budget is None:
        return
    unpriced = unpriced_labels(estimates)
    if unpriced:
        raise InvalidInput(
            f"--budget cannot be checked: no listed price for {', '.join(unpriced)}",
            hint="Add a prices table for that model to targets.toml, or drop --budget",
        )
    total = estimate_total(estimates, pre_check=pre_check)
    # the bound ranks by input price while the total also prices generation, so the
    # ceiling is whichever of the two is higher, never a bound that undercuts the total
    ceiling = total if upper_bound is None else max(upper_bound.usd, total or 0.0)
    if ceiling is None or ceiling <= budget:
        return
    if total is None:
        message = f"--budget ${budget:.4f} cannot cover what the run can cost"
    elif total > budget:
        message = f"The worst-case estimate ${total:.4f} exceeds --budget ${budget:.4f}"
    else:
        message = (
            f"The worst-case estimate ${total:.4f} fits --budget ${budget:.4f}, but the run "
            f"can cost more"
        )
    if (
        upper_bound is not None
        and (total is None or upper_bound.usd > total)
        and not same_amount(upper_bound.usd, total)
    ):
        message += (
            f"; the upper bound is ${upper_bound.usd:.4f} if the {upper_bound.keep} priciest "
            f"candidates are the ones that answer"
        )
    elif (
        pre_check is not None
        and pre_check.usd is not None
        and total is not None
        and (total - pre_check.usd <= budget)
    ):
        # the specs alone fit: the availability check is what tips the total over
        message += f", ${pre_check.usd:.4f} of it the availability check"
    raise InvalidInput(message, hint=hint)


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
