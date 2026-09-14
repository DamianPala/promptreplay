"""Planning a sweep: which endpoints it will probe, checked how, and the record of that.

`commands/sweep.py` owns the command — the flags, the flow, the document it prints.
Everything the flow needs to decide *which* specs it runs lives here, because each piece
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
    exclude: Sequence[str]
    sort: str
    top: int | None
    zdr: bool
    check: bool
    """The availability check is planned: `--top` turns it on, `--check` adds it alone."""


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
        sort=criteria.sort,
        top=criteria.top,
        zdr=criteria.zdr,
        check=criteria.check,
        dropped=list(selection.dropped),
        ranking=[ranked(endpoint) for endpoint in selection.kept],
    )


def select(
    model: str,
    gateway: Target,
    endpoints: Sequence[Endpoint],
    *,
    include: Sequence[str],
    exclude: Sequence[str],
    sort: str,
    zdr: bool,
) -> Selection:
    """Apply the tag filters, the percentile requirement, ZDR and the stability floor."""
    from provibench.bench.selection import select_candidates

    filtered = filtered_endpoints(model, endpoints, include=include, exclude=exclude)
    require_percentiles(filtered, sort, gateway)
    return select_candidates(
        filtered,
        gateway=gateway,
        model=model,
        include=include,
        sort=sort,
        zdr=zdr,
        zdr_tags=fetch_zdr_tags(model) if zdr else frozenset(),
    )


def endpoint_list(
    model: str, *, api_key: str | None = None
) -> tuple[dict[str, list[Endpoint]], list[Endpoint]]:
    """The model's OpenRouter endpoints: the run's snapshot and the list selection reads.

    A sweep without that list is not a sweep, so a failed or empty lookup fails the command
    here — before a spec is built, an estimate printed or an API key asked for.
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
