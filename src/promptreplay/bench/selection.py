"""Choosing which endpoints a sweep probes: the stability floor, the sort key, ZDR, `--top`.

A sweep's listing is a pre-filter, not a verdict: the run measures the endpoints it picked
and `report` orders them by the effective price it measured. What this module decides is
which endpoints are worth paying to measure at all — one serving a degraded status or a bad
day of uptime cannot produce a number comparable with the rest, and one the account's
settings exclude cannot be probed at all — and it owns the record of that decision
(`SweepInfo`), since `report` shows the selection offline, with no network.

The human view of that decision — the listing before a run and the sentence a report reads
back — lives in `bench.selection_text`, split out for this module's line budget; the criteria
and the record they leave stay here.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import AliasChoices, BaseModel, Field

from promptreplay.bench.openrouter import Endpoint
from promptreplay.bench.targets import RunSpec, Target
from promptreplay.core.errors import InvalidInput

if TYPE_CHECKING:
    from promptreplay.core.documents import Document

__all__ = [
    "PERCENTILE_SORTS",
    "SORT_KEYS",
    "UPTIME_FLOOR",
    "RankedEndpoint",
    "Selection",
    "SelectionCriteria",
    "SelectionDrop",
    "SweepInfo",
    "missing_percentiles",
    "pinned_spec",
    "prefixed",
    "rank_candidates",
    "ranked",
    "select_candidates",
    "sort_key",
    "stability_reason",
]

type SortKey = Literal["price", "uptime", "throughput", "latency"]

SORT_KEYS: tuple[SortKey, ...] = ("price", "uptime", "throughput", "latency")
"""Every `--sort` key, in the order help lists them; `price` is the default."""

PERCENTILE_SORTS: Set[str] = frozenset({"throughput", "latency"})
"""The sorts that rank by a 30-minute percentile: OpenRouter returns those only with a key."""

UPTIME_FLOOR = 97.0
"""Default percent of the last day below which an endpoint is having a bad day, not a
candidate: `sweep --min-uptime` overrides it per run, and this is that flag's own default."""

_SUPPORTED_SORTS: Set[str] = frozenset(SORT_KEYS)


class RankedEndpoint(BaseModel):
    """One candidate as the ranking read it, so the record says what the order was made of."""

    tag: str
    price_input: float
    uptime_1d: float | None = None
    status: int | str | None = None
    latency_p50_ms: float | None = None
    throughput_p50_tok_s: float | None = None


class SelectionDrop(BaseModel):
    """One endpoint the sweep did not probe, and why."""

    tag: str
    endpoint: str = Field(validation_alias=AliasChoices("endpoint", "spec"))
    """The label of the endpoint; `spec` is the key a run written before the rename used."""
    reason: str
    checked: bool = False
    """The availability pre-check produced the reason, not the listing criteria."""

    @classmethod
    def for_spec(cls, spec: RunSpec, reason: str, *, checked: bool = False) -> SelectionDrop:
        """The drop of one endpoint, tagged the way that endpoint pins its provider."""
        return cls(
            tag=spec.providers[0] if spec.providers else spec.model,
            endpoint=spec.label,
            reason=reason,
            checked=checked,
        )


class SweepInfo(BaseModel):
    """What a sweep covered: the model, the criteria, and what they chose.

    Recorded so a reader of one run directory can tell how the endpoints were chosen — a
    sweep with `--top 5 --sort uptime` is a different measurement from the same command
    without the criteria, and the endpoint list alone does not say which of them produced it.
    `ranking` is the snapshot the order was decided from, so `report` prints the selection
    offline; `dropped` names the endpoints the criteria removed, and why.
    """

    model: str
    target: str
    included: list[str] = Field(default_factory=list[str])
    excluded: list[str] = Field(default_factory=list[str])
    quantization: list[str] = Field(default_factory=list[str])
    pinned: list[str] = Field(default_factory=list[str])
    sort: str = "price"
    top: int | None = None
    zdr: bool = False
    uptime_floor: float = UPTIME_FLOOR
    """The `--min-uptime` this run used; defaults to `UPTIME_FLOOR` so a run recorded before
    the flag existed still reads back the floor it was actually held to."""
    check: bool = False
    """The run checks availability: `--top` turns it on, `--check` adds it on its own."""
    dropped: list[SelectionDrop] = Field(default_factory=list[SelectionDrop])
    ranking: list[RankedEndpoint] = Field(default_factory=list[RankedEndpoint])

    def to_document(self) -> Document:
        """The block as a command's output carries it: `status` reads as a string there.

        The API sends a status as a number or as a string, and the record keeps whatever it
        got; the document schema has one type per field, so a number becomes its own text —
        the same thing `promptreplay endpoints` prints.
        """
        block = self.model_dump()
        block["ranking"] = [
            {**ranked, "status": None if ranked["status"] is None else str(ranked["status"])}
            for ranked in block["ranking"]
        ]
        return block


@dataclass(frozen=True, slots=True)
class Selection:
    """What the criteria left: the candidates in ranking order, and the ones they removed."""

    kept: list[Endpoint]
    dropped: list[SelectionDrop]


def pinned_spec(gateway: Target, model: str, tag: str) -> RunSpec:
    """The run spec that probes one endpoint: the gateway's model pinned to that tag."""
    return RunSpec(target=gateway, model=model, providers=[tag])


@dataclass(frozen=True, slots=True)
class SelectionCriteria:
    """The tag and health axes `select_candidates` filters and ranks by.

    Bundled into one object because `gateway` and `model` — needed to label a drop with the
    endpoint it removes — pushed the flat keyword signature past the project's argument
    limit once quantization and pin joined `include`, `sort`, `zdr` and the floor.
    """

    include: Sequence[str] = ()
    pin: Sequence[str] = ()
    sort: str = "price"
    zdr: bool = False
    zdr_tags: Set[str] = frozenset()
    uptime_floor: float = UPTIME_FLOOR
    quantization: Sequence[str] = ()


_DEFAULT_CRITERIA = SelectionCriteria()


def select_candidates(
    endpoints: Sequence[Endpoint],
    *,
    gateway: Target,
    model: str,
    criteria: SelectionCriteria = _DEFAULT_CRITERIA,
) -> Selection:
    """Apply the quantization filter, the ZDR intersection and the stability floor, then rank.

    The order is the one the listing explains: quantization and ZDR first (an endpoint of a
    quantization the caller did not ask for, or outside the account's ZDR list, is not a
    candidate at all regardless of any tag filter), then the floor, then the sort. `--include`
    and `--pin` both override the floor for the tag they name, but neither bypasses
    quantization or ZDR: naming a tag is the operator asking for one axis at a time, not for
    every axis to be skipped.

    A pinned tag is exempt from `--top` too, which this function expresses in the order it
    returns: `kept` ranks the non-pinned survivors by `sort` and appends the pinned survivors
    after them, in the order `pin` named them, so a caller who stops after N candidates never
    cuts a pin. The tag filters are the caller's: an `--include` tag no endpoint has is an
    input error, not a drop, and so is a `--pin` tag no endpoint has.
    """
    allowed = set(criteria.zdr_tags)
    pin_set = set(criteria.pin)
    wanted_quant = {value.lower() for value in criteria.quantization}
    _check_pins_exist(endpoints, criteria.pin)

    def drop(endpoint: Endpoint, reason: str) -> SelectionDrop:
        return SelectionDrop.for_spec(pinned_spec(gateway, model, endpoint.tag), reason)

    kept: list[Endpoint] = []
    pinned_kept: list[Endpoint] = []
    dropped: list[SelectionDrop] = []
    for endpoint in endpoints:
        if wanted_quant:
            quant_reason = _quantization_reason(endpoint, wanted_quant)
            if quant_reason is not None:
                dropped.append(drop(endpoint, quant_reason))
                continue
        if criteria.zdr and endpoint.tag not in allowed:
            dropped.append(drop(endpoint, "not ZDR"))
            continue
        is_pinned = endpoint.tag in pin_set
        reason = stability_reason(
            endpoint,
            included=prefixed(endpoint.tag, criteria.include) or is_pinned,
            floor=criteria.uptime_floor,
        )
        if reason is not None:
            dropped.append(drop(endpoint, reason))
            continue
        (pinned_kept if is_pinned else kept).append(endpoint)
    pin_order = {tag: index for index, tag in enumerate(dict.fromkeys(criteria.pin))}
    pinned_kept.sort(key=lambda endpoint: pin_order[endpoint.tag])
    return Selection(kept=[*rank_candidates(kept, criteria.sort), *pinned_kept], dropped=dropped)


def _check_pins_exist(endpoints: Sequence[Endpoint], pin: Sequence[str]) -> None:
    """A `--pin` tag no endpoint has is a usage error, the same one `--include` raises."""
    tags = {endpoint.tag for endpoint in endpoints}
    for tag in dict.fromkeys(pin):
        if tag not in tags:
            available = ", ".join(sorted(tags)) or "(none)"
            raise InvalidInput(
                f"--pin {tag!r} matches no endpoint; available tags: {available}",
                hint="Tags are the endpoint slugs that `promptreplay endpoints MODEL` prints",
            )


def _quantization_reason(endpoint: Endpoint, wanted: Set[str]) -> str | None:
    """Why `--quantization` drops this endpoint, or `None` when its quantization is wanted.

    `unknown` in `wanted` matches an endpoint the listing does not label; the comparison is
    case-insensitive because OpenRouter's own casing for a quantization is not consistent
    (`fp8` and `FP8` both appear).
    """
    label = (endpoint.quantization or "unknown").lower()
    return None if label in wanted else f"quantization {label}"


def stability_reason(
    endpoint: Endpoint, *, included: bool, floor: float = UPTIME_FLOOR
) -> str | None:
    """Why the stability floor drops this endpoint, or `None` when it passes.

    `--include` overrides the floor: naming a tag is the operator saying they want that
    endpoint measured anyway, which is the one way back in for a degraded endpoint;
    `--pin` overrides it the same way, for the exact tag it names.

    The dropped uptime is printed to two decimals because one is not enough to explain the
    drop: a value just under the floor rounds up to the floor itself (`96.96` reads `97.0`),
    which makes the reason look like it contradicts the criterion it states.
    """
    if included:
        return None
    status = _status_number(endpoint.status)
    if status is not None and status < 0:
        return f"status {endpoint.status}"
    uptime = endpoint.uptime_1d
    if uptime is not None and uptime < floor:
        return f"uptime 1d {uptime:.2f} %"
    return None


def rank_candidates(endpoints: Sequence[Endpoint], sort: str) -> list[Endpoint]:
    """The candidates in `--sort` order; an endpoint with no percentile sorts last.

    The order is stable, so endpoints a key cannot tell apart keep the order the API listed
    them in, which makes two sweeps of one model comparable.
    """
    return sorted(endpoints, key=lambda endpoint: sort_key(endpoint, sort))


def sort_key(endpoint: Endpoint, sort: str) -> tuple[object, ...]:
    """The ranking key: the sort's own field first, then the ties help documents.

    `price` breaks its ties by throughput p50 descending and then uptime 1d descending,
    both `when present`: an endpoint with no throughput percentile is not a throughput
    winner, it is an unknown, so it sorts after the endpoints that have one.
    """
    if sort not in _SUPPORTED_SORTS:
        raise ValueError(f"unknown sort key {sort!r}; expected one of {', '.join(SORT_KEYS)}")
    if sort == "uptime":
        return (_descending(endpoint.uptime_1d), endpoint.prices.input)
    if sort == "throughput":
        return (_descending(endpoint.throughput_30m),)
    if sort == "latency":
        return (_ascending(endpoint.latency_ms_30m),)
    return (
        endpoint.prices.input,
        _descending(endpoint.throughput_30m),
        _descending(endpoint.uptime_1d),
    )


def missing_percentiles(endpoints: Sequence[Endpoint], sort: str) -> list[str]:
    """The tags the sort key has no value for: the percentiles a keyless fetch leaves null."""
    field = "throughput_30m" if sort == "throughput" else "latency_ms_30m"
    return [endpoint.tag for endpoint in endpoints if getattr(endpoint, field) is None]


def ranked(endpoint: Endpoint) -> RankedEndpoint:
    """One candidate as the run's record snapshots it."""
    return RankedEndpoint(
        tag=endpoint.tag,
        price_input=endpoint.prices.input,
        uptime_1d=endpoint.uptime_1d,
        status=endpoint.status,
        latency_p50_ms=endpoint.latency_ms_30m,
        throughput_p50_tok_s=endpoint.throughput_30m,
    )


def prefixed(tag: str, include: Sequence[str]) -> bool:
    """Whether a tag is one the caller named with `--include`, which overrides the floor.

    Public because `selection_text.selection_line` reads it back too, to count how many of
    the run's kept endpoints an `--include` prefix actually named.
    """
    return any(tag.startswith(prefix) for prefix in include)


def _descending(value: float | None) -> tuple[bool, float]:
    """Higher first, unknown last: the flag keeps a missing value from outranking a real one."""
    return (value is None, -(value or 0.0))


def _ascending(value: float | None) -> tuple[bool, float]:
    return (value is None, value or 0.0)


def _status_number(status: int | str | None) -> int | None:
    """A status as a number, for the ones the API sends as strings; `None` when unreadable."""
    if isinstance(status, bool) or status is None:
        return None
    if isinstance(status, int):
        return status
    try:
        return int(status)
    except ValueError:
        return None
