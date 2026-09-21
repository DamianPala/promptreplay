"""Planning a sweep: which endpoints it will probe, checked how, and the record of that.

`commands/sweep.py` owns the command — the flags, the flow, the document it prints.
Everything the flow needs to decide *which* endpoints it runs lives here, because each piece
answers a question the command's help has to keep straight: the tag filters and the
percentile requirement (what the list can even be ranked by), the API key that makes the
endpoint list carry the health it ranks by, the ZDR intersection and the stability floor
(`bench.selection`), and the `sweep` block that records the whole decision so `report` can
print it offline. The availability check itself is a phase of the run, so it lives with the
run (`commands.probe`), which is also where its cost is priced.

`provibench.bench.*` pulls in httpx and pydantic; its symbols are imported only inside the
functions that use them, so building the CLI (schema, --help, completion) stays cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from provibench.commands.run_specs import (
    fetch_model_endpoints,
    fetch_zdr_tags,
    filtered_endpoints,
)
from provibench.core.context import Invocation
from provibench.core.errors import InvalidInput, OperationFailed

if TYPE_CHECKING:
    from provibench.bench.openrouter import Endpoint
    from provibench.bench.selection import Selection, SweepInfo
    from provibench.bench.targets import Target


@dataclass(frozen=True, slots=True)
class Criteria:
    """The selection criteria the run was told to use, as the record keeps them."""

    model: str
    gateway: Target
    include: Sequence[str]
    pin: tuple[str, ...]
    exclude: Sequence[str]
    sort: str
    top: int | None
    zdr: bool
    uptime_floor: float
    check: bool
    """The availability check is planned: `--top` turns it on, `--check` adds it alone."""
    quantization: tuple[str, ...]


def sweep_info(selection: Selection, criteria: Criteria) -> SweepInfo:
    """The run's record of what it chose: the criteria, the ranking, and what it dropped.

    The check's verdicts are added by the run itself, once it has visited the candidates.
    """
    from provibench.bench.selection import SweepInfo, ranked

    return SweepInfo(
        model=criteria.model,
        target=criteria.gateway.name,
        included=list(criteria.include),
        excluded=list(criteria.exclude),
        quantization=list(criteria.quantization),
        pinned=list(criteria.pin),
        sort=criteria.sort,
        top=criteria.top,
        zdr=criteria.zdr,
        uptime_floor=criteria.uptime_floor,
        check=criteria.check,
        dropped=list(selection.dropped),
        ranking=[ranked(endpoint) for endpoint in selection.kept],
    )


def select(endpoints: Sequence[Endpoint], criteria: Criteria) -> Selection:
    """Apply the tag filters, the percentile requirement, ZDR and the stability floor.

    Takes the whole `Criteria` rather than one keyword per field: quantization and pin would
    have pushed a flat signature (already carrying `model`, `gateway`, `include`, `exclude`,
    `sort`, `zdr`, `uptime_floor`) past the project's own argument limit.
    """
    from provibench.bench.selection import SelectionCriteria, select_candidates

    model, gateway = criteria.model, criteria.gateway
    _reject_excluded_pins(criteria)
    filtered = filtered_endpoints(
        model, endpoints, include=criteria.include, exclude=criteria.exclude, pin=criteria.pin
    )
    require_percentiles(filtered, criteria.sort, gateway)
    return select_candidates(
        filtered,
        gateway=gateway,
        model=model,
        criteria=SelectionCriteria(
            include=criteria.include,
            pin=criteria.pin,
            sort=criteria.sort,
            zdr=criteria.zdr,
            zdr_tags=fetch_zdr_tags(model) if criteria.zdr else frozenset(),
            uptime_floor=criteria.uptime_floor,
            quantization=criteria.quantization,
        ),
    )


def _reject_excluded_pins(criteria: Criteria) -> None:
    """A tag that is both pinned and excluded is a contradiction, not an unknown tag.

    `--exclude` cuts the listing before the pin is looked up, so without this the caller who
    named the same tag twice is told it matches no endpoint -- true of the filtered list, and
    unanswerable next to the tag they can see in `provibench endpoints MODEL`.
    """
    for tag in criteria.pin:
        prefix = next((value for value in criteria.exclude if tag.startswith(value)), None)
        if prefix is not None:
            raise InvalidInput(
                f"--pin {tag!r} is also excluded by --exclude {prefix!r}",
                hint="Drop one of the two: a pinned endpoint is always probed",
            )


def endpoint_list(
    model: str, *, api_key: str | None = None
) -> tuple[dict[str, list[Endpoint]], list[Endpoint]]:
    """The model's OpenRouter endpoints: the run's snapshot and the list selection reads.

    A sweep without that list is not a sweep, so a failed or empty lookup fails the command
    here — before an endpoint is built, an estimate printed or an API key asked for.
    """
    index, notes = fetch_model_endpoints([model], api_key=api_key)
    endpoints = index.get(model, [])
    if not endpoints:
        raise OperationFailed(
            notes[0] if notes else f"No OpenRouter endpoint serves the model {model!r}",
            hint="Check the slug with: provibench endpoints MODEL",
        )
    return index, endpoints


def gateway_key(sort: str, gateway: Target, invocation: Invocation) -> str | None:
    """The gateway's API key for the endpoint fetch, or `invalid_input` when it is needed.

    OpenRouter returns the health percentiles only for a request that carries the key, so the
    list is fetched with it whenever the target has one — under the default `--sort price` the
    listing's p50 columns and the documented throughput tie-break depend on it too. A sort
    that ranks *by* a percentile cannot fall back on anything else, so the same missing key is
    an input error there, naming the flag and the variable to set.
    """
    from provibench.bench.selection import PERCENTILE_SORTS
    from provibench.bench.targets import resolve_api_key

    if sort not in PERCENTILE_SORTS:
        return invocation.env.get(gateway.api_key_env) or None
    try:
        return resolve_api_key(gateway, invocation.env)
    except ValueError as exc:
        raise InvalidInput(
            f"--sort {sort} ranks by each endpoint's 30-minute percentiles, which "
            f"OpenRouter returns only when the request carries the target's API key: {exc}",
            hint="Set that key, or sort by price or uptime",
        ) from exc


def require_percentiles(endpoints: Sequence[Endpoint], sort: str, gateway: Target) -> None:
    """Fail when a candidate has no percentile to rank by, rather than call it the worst.

    A key can be set and still return nothing for an endpoint — no traffic in the last 30
    minutes — and sorting an unknown last is a claim the data does not support.
    """
    from provibench.bench.selection import PERCENTILE_SORTS, missing_percentiles

    if sort not in PERCENTILE_SORTS:
        return
    missing = missing_percentiles(endpoints, sort)
    if not missing:
        return
    raise InvalidInput(
        f"--sort {sort}: OpenRouter returned no {sort}_last_30m for {len(missing)} of "
        f"{len(endpoints)} endpoint(s) with {gateway.api_key_env}: {', '.join(missing)}",
        hint="Exclude them with --exclude, or sort by price or uptime",
    )
