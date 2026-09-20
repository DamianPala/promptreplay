"""Choosing which endpoints a sweep probes: the stability floor, the sort key, ZDR, `--top`.

A sweep's listing is a pre-filter, not a verdict: the run measures the endpoints it picked
and `report` orders them by the effective price it measured. What this module decides is
which endpoints are worth paying to measure at all — one serving a degraded status or a bad
day of uptime cannot produce a number comparable with the rest, and one the account's
settings exclude cannot be probed at all — and it owns the record of that decision
(`SweepInfo`), since `report` shows the selection offline, with no network.

The listing is the human view of the same decision, so the criteria the run chose by and the
endpoints they removed are named in one place: `render_candidates` for the command that is
about to spend money, `selection_line` for the report of a run that already did.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import AliasChoices, BaseModel, Field

from provibench.bench.labels import join_and, text_table
from provibench.bench.openrouter import Endpoint
from provibench.bench.targets import RunSpec, Target

if TYPE_CHECKING:
    from provibench.core.documents import Document

__all__ = [
    "PERCENTILE_SORTS",
    "SORT_KEYS",
    "UPTIME_FLOOR",
    "RankedEndpoint",
    "Selection",
    "SelectionDrop",
    "SweepInfo",
    "missing_percentiles",
    "not_probed_lines",
    "pinned_spec",
    "rank_candidates",
    "ranked",
    "render_candidates",
    "select_candidates",
    "selection_line",
    "sort_key",
    "stability_reason",
]

type SortKey = Literal["price", "uptime", "throughput", "latency"]

SORT_KEYS: tuple[SortKey, ...] = ("price", "uptime", "throughput", "latency")
"""Every `--sort` key, in the order help lists them; `price` is the default."""

PERCENTILE_SORTS: Set[str] = frozenset({"throughput", "latency"})
"""The sorts that rank by a 30-minute percentile: OpenRouter returns those only with a key."""

UPTIME_FLOOR = 97.0
"""Percent of the last day below which an endpoint is having a bad day, not a candidate."""

_SUPPORTED_SORTS: Set[str] = frozenset(SORT_KEYS)

_FLOOR_MARGIN = 0.1
"""How close to the floor an uptime has to be before the table prints it to two decimals."""

_CANDIDATE_COLUMNS = ("tag", "status", "uptime 1d", "$/M in", "lat p50 ms", "tput p50")


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
    sort: str = "price"
    top: int | None = None
    zdr: bool = False
    check: bool = False
    """The run checks availability: `--top` turns it on, `--check` adds it on its own."""
    dropped: list[SelectionDrop] = Field(default_factory=list[SelectionDrop])
    ranking: list[RankedEndpoint] = Field(default_factory=list[RankedEndpoint])

    def to_document(self) -> Document:
        """The block as a command's output carries it: `status` reads as a string there.

        The API sends a status as a number or as a string, and the record keeps whatever it
        got; the document schema has one type per field, so a number becomes its own text —
        the same thing `provibench endpoints` prints.
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


def select_candidates(
    endpoints: Sequence[Endpoint],
    *,
    gateway: Target,
    model: str,
    include: Sequence[str] = (),
    sort: str = "price",
    zdr: bool = False,
    zdr_tags: Set[str] = frozenset(),
) -> Selection:
    """Apply the ZDR intersection and the stability floor, then rank what survives.

    The order is the one the listing explains: ZDR first (an endpoint outside the account's
    list is not available on the account's terms at all), then the floor, then the sort.
    The tag filters are the caller's: a tag no endpoint has is an input error, not a drop.
    """
    allowed = set(zdr_tags)

    def drop(endpoint: Endpoint, reason: str) -> SelectionDrop:
        return SelectionDrop.for_spec(pinned_spec(gateway, model, endpoint.tag), reason)

    kept: list[Endpoint] = []
    dropped: list[SelectionDrop] = []
    for endpoint in endpoints:
        if zdr and endpoint.tag not in allowed:
            dropped.append(drop(endpoint, "not ZDR"))
            continue
        reason = stability_reason(endpoint, included=_prefixed(endpoint.tag, include))
        if reason is not None:
            dropped.append(drop(endpoint, reason))
            continue
        kept.append(endpoint)
    return Selection(kept=rank_candidates(kept, sort), dropped=dropped)


def stability_reason(endpoint: Endpoint, *, included: bool) -> str | None:
    """Why the stability floor drops this endpoint, or `None` when it passes.

    `--include` overrides the floor: naming a tag is the operator saying they want that
    endpoint measured anyway, which is the one way back in for a degraded endpoint.

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
    if uptime is not None and uptime < UPTIME_FLOOR:
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


def render_candidates(selection: Selection, *, model: str, sort: str, zdr: bool) -> list[str]:
    """The listing printed before the estimate: the candidates, then what the criteria cut."""
    kept = len(selection.kept)
    header = f"candidates: {model}, {kept} endpoint(s), sort={sort}, zdr={_on_off(zdr)}"
    rows = [_candidate_row(endpoint) for endpoint in selection.kept]
    lines = [header, *text_table(_CANDIDATE_COLUMNS, rows)]
    if selection.dropped:
        lines.append("dropped: " + ", ".join(_drop_cell(drop) for drop in selection.dropped))
    return lines


def selection_line(sweep: SweepInfo) -> str:
    """A sentence: how many endpoints the run took and by what, then why the listing itself
    dropped any candidate (the stability floor or the ZDR filter, in the reader's words).

    A pre-check's own drops are `not_probed_lines`'s job, kept separate because a pre-check
    spends money to find them and a listing-time drop costs nothing.
    """
    if sweep.top is None:
        which = f"every candidate by {sweep.sort}"
    elif sweep.sort == "price":
        which = f"the {sweep.top} cheapest by listed price"
    else:
        which = f"the {sweep.top} best by {sweep.sort}"
    line = f"Of the OpenRouter providers, the run took {which}"
    if sweep.zdr:
        line += ", zero-data-retention endpoints only"
    line += "."
    sentences = _drop_sentences(drop for drop in sweep.dropped if not drop.checked)
    if sentences:
        line += " " + " ".join(sentences)
    return line


def not_probed_lines(sweep: SweepInfo) -> list[str]:
    """One sentence per reason the availability pre-check skipped a candidate.

    Grouped the same way `selection_line` groups its own drops, but a pre-check failure
    keeps its own gateway reason rather than one of the listing's short codes.
    """
    return _drop_sentences(drop for drop in sweep.dropped if drop.checked)


def _drop_sentences(dropped: Iterable[SelectionDrop]) -> list[str]:
    """`{label} was skipped because {reason}.`, endpoints sharing a reason joined into one
    sentence, named the way the table above it does (`_short_drop_label`)."""
    groups: dict[str, list[str]] = {}
    for drop in dropped:
        groups.setdefault(_reason_words(drop), []).append(_short_drop_label(drop))
    sentences: list[str] = []
    for reason, labels in groups.items():
        verb = "was skipped" if len(labels) == 1 else "were skipped"
        sentences.append(f"{join_and(labels)} {verb} because {reason}.")
    return sentences


def _short_drop_label(drop: SelectionDrop) -> str:
    """`@tag` when the label has a provider suffix, else the label whole, as the table reads."""
    return f"@{drop.tag}" if "@" in drop.endpoint else drop.endpoint


def _reason_words(drop: SelectionDrop) -> str:
    """A drop's own code (`status -2`, `uptime 1d 77.90 %`) in the reader's words.

    A pre-check failure already carries its provider's own gateway reason -- that is the
    "words" a reader can act on, and rewriting it would replace one true statement with a
    guess. A status code is left out: the `candidates:` listing and the run file keep it.
    """
    if drop.checked:
        return drop.reason
    if drop.reason.startswith("status "):
        return "OpenRouter reported it as degraded"
    if drop.reason.startswith("uptime 1d "):
        pct = drop.reason.removeprefix("uptime 1d ").strip()
        return f"its one-day uptime was below the floor ({pct})"
    if drop.reason == "not ZDR":
        return "it is not a zero-data-retention endpoint"
    return drop.reason


def _drop_cell(drop: SelectionDrop) -> str:
    """One dropped endpoint in the listing's own notation: `tag (reason)`."""
    return f"{drop.tag} ({drop.reason})"


def _candidate_row(endpoint: Endpoint) -> list[str]:
    return [
        endpoint.tag,
        "-" if endpoint.status is None else str(endpoint.status),
        _uptime_cell(endpoint.uptime_1d),
        f"{endpoint.prices.input:.3f}",
        _number(endpoint.latency_ms_30m, digits=0),
        _number(endpoint.throughput_30m, digits=1),
    ]


def _uptime_cell(value: float | None) -> str:
    """The uptime column: one decimal, or two when the value sits next to the floor.

    `97.0` in this column could be anything from 96.95 up, and an endpoint kept by
    `--include` would then be printed as a value the floor should have dropped. The second
    digit is only paid where it disambiguates; a healthy endpoint reads as it always did.
    """
    if value is None:
        return "-"
    return f"{value:.{_uptime_digits(value)}f}"


def _uptime_digits(value: float) -> int:
    """How many decimals an uptime needs: two within `_FLOOR_MARGIN` of the floor, else one."""
    return 2 if abs(value - UPTIME_FLOOR) < _FLOOR_MARGIN else 1


def _number(value: float | None, *, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _on_off(flag: bool) -> str:
    return "on" if flag else "off"


def _prefixed(tag: str, include: Sequence[str]) -> bool:
    """Whether a tag is one the caller named with `--include`, which overrides the floor."""
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
