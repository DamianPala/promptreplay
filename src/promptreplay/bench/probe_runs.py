"""Persisting and reading probe runs: `run.json` plus one `<slug>.jsonl` per spec.

A run directory is the record of what was spent, so it carries everything needed to
re-read the result offline: the rungs and repeats that ran, the endpoint facts at run time,
and the price of each spec. `report` never needs a network.

The snapshot (`endpoints`, one record per spec) is the part of that record that goes out
of date: the endpoint's quantization, context length, uptime and status as the run's own
listing reported them. `history` and `compare` read it alongside `prices`; a run written
before this block existed carries none, and still gets its listed price from `prices`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, model_validator

from promptreplay.bench.estimate import SpecPrices, match_endpoint
from promptreplay.bench.openrouter import Endpoint, normalize_provider
from promptreplay.bench.openrouter_stats import ReportedAverage
from promptreplay.bench.probe_models import ProbeOptions, ProbeResult, ProbeRun
from promptreplay.bench.replay import RunRef
from promptreplay.bench.selection import SweepInfo
from promptreplay.bench.spend import precheck_spend, precheck_worst_case
from promptreplay.bench.targets import Prices, RunSpec

__all__ = [
    "EndpointSnapshot",
    "PrecheckSummary",
    "ProbeRunMeta",
    "SweepInfo",
    "endpoint_snapshot",
    "listing_prices",
    "load_probe_run",
    "read_protocol",
    "write_probe_run",
]

"""`SweepInfo` lives in `bench.selection` with the criteria it records; it is re-exported
here because a run directory is where a reader meets it."""

_PRECHECK_FILE = "precheck.jsonl"
"""The availability check's own requests, kept out of every spec's `<slug>.jsonl`."""


class EndpointSnapshot(BaseModel):
    """One spec's endpoint as the run's own listing described it: the facts that drift.

    A gateway spec's record is the endpoint it pinned (`@tag`): quantization, context
    length, 1-day uptime and status come from the endpoint list. A native spec has no
    endpoint to pin. Everything is nullable because a listing may omit any of it, and
    because a native spec has nothing to say about endpoint facts.
    """

    tag: str | None = None
    """The pinned endpoint's tag; `None` for a native spec and for an unpinned gateway one."""
    quantization: str | None = None
    context_length: int | None = None
    uptime_1d: float | None = None
    status: int | str | None = None


class PrecheckSummary(BaseModel):
    """What the availability check spent, folded to the two numbers a reader wants.

    The requests themselves are in `precheck.jsonl`, role `precheck`, kept out of every
    spec's own hit/error/drift statistics; this is only their count, their price and their
    worst case.
    """

    requests: int
    spend_usd: float | None = None
    worst_case_usd: float | None = None
    """What the pre-check's own requests would have cost with no cache hit; `None` for a run
    written before this field existed, or when nothing in the check had a listed price."""


class ProbeRunMeta(BaseModel):
    """The `run.json` of a persisted probe run."""

    protocol: Literal["probe"] = "probe"
    trace: str
    conversation: str
    created: str
    run_hex: str
    options: ProbeOptions
    specs: list[RunRef]
    ttl_s: list[int] | None = None
    """The offsets `--ttl` re-read the first rung at; `None` when TTL never ran."""
    endpoints: dict[str, EndpointSnapshot] = Field(default_factory=dict)
    """One snapshot per spec label; empty for a run that stored no listing."""
    prices: dict[str, SpecPrices] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    sweep: SweepInfo | None = None
    """The sweep this run expanded from; `None` for a run whose specs were given by hand."""
    precheck: PrecheckSummary | None = None
    """`None` for a run with no availability check, or one written before this field existed."""
    trace_prompt_tokens: int | None = None
    """The trace's total prompt tokens (`bench.trace.total_prompt_tokens`), for
    `session_prompt_usd`; `None` for a run written before this field existed."""
    listing_prices: dict[str, Prices] | None = None
    """The swept or probed gateway model's own endpoint listing, keyed by normalized
    provider name (`bench.openrouter.normalize_provider`); an unpinned spec is repriced by
    looking its served provider up here. `None` (not `{}`) for a run written before this
    field existed, so `choose_prices` knows to fit the served rate instead; `{}` for a run
    with no OpenRouter spec to list."""
    reported_average: ReportedAverage | None = None
    """OpenRouter's reported cache share for the last complete UTC day, fetched once for the
    sweep's or the first pinned spec's model; `None` for a run written before this field
    existed, one with no pinned OpenRouter spec, or one where the fetch failed."""

    @model_validator(mode="before")
    @classmethod
    def _drop_pre_slice_listing(cls, data: object) -> object:
        """Skip the whole endpoint listing a run written before this slice stored here.

        That block was `{model slug: [endpoint, ...]}` — the gateway's full list, not a
        per-spec snapshot — so the entries that are lists are dropped on load. The file
        keeps them, and the reader shows `-` for those runs, which is the honest answer:
        the run did not record what its endpoints said about themselves. A snapshot is an
        object (or an `EndpointSnapshot` on the way in), so this cannot drop one.
        """
        document = _as_mapping(data)
        raw = _as_mapping(document.get("endpoints")) if document is not None else None
        if document is None or raw is None:
            return data
        kept = {label: value for label, value in raw.items() if not isinstance(value, list)}
        if len(kept) == len(raw):
            return data
        return {**document, "endpoints": kept}


def endpoint_snapshot(
    specs: Sequence[RunSpec],
    index: Mapping[str, list[Endpoint]],
) -> dict[str, EndpointSnapshot]:
    """One snapshot per spec: the endpoint facts that are unique to this record.

    The endpoint facts come from the listing the run already fetched; a spec that pinned
    no tag has none. Prices remain in the run's `prices` block, which is the source shared
    by `report`, `history` and `compare`.
    """
    records: dict[str, EndpointSnapshot] = {}
    for spec in specs:
        # only a pinned spec has one endpoint to describe; an unpinned one may have been
        # served by any of them, so it records no endpoint facts (the matcher would hand
        # back the priciest endpoint, which is the estimate's worst case, not this run's)
        endpoint = match_endpoint(spec, index.get(spec.model, [])) if spec.providers else None
        records[spec.label] = EndpointSnapshot(
            tag=endpoint.tag if endpoint is not None else _pinned_tag(spec),
            quantization=endpoint.quantization if endpoint is not None else None,
            context_length=endpoint.context_length if endpoint is not None else None,
            uptime_1d=endpoint.uptime_1d if endpoint is not None else None,
            status=endpoint.status if endpoint is not None else None,
        )
    return records


def listing_prices(
    specs: Sequence[RunSpec],
    index: Mapping[str, list[Endpoint]],
) -> dict[str, Prices]:
    """Every OpenRouter endpoint's listed prices, keyed by its normalized provider name.

    Persisted once per run so `bench.probe_pricing.choose_prices` can reprice an unpinned
    spec by what actually served it without a second, later lookup; a native spec's model
    has no entry in `index` and contributes nothing. Two endpoints of the same provider at
    different quantizations collide on the same key -- the last one in the listing wins,
    the same ambiguity an unpinned spec already carries (it does not say which quantization
    served it either).
    """
    prices: dict[str, Prices] = {}
    for spec in specs:
        if spec.target.kind != "openrouter":
            continue
        for endpoint in index.get(spec.model, []):
            key = normalize_provider(endpoint.provider_name or endpoint.tag)
            prices[key] = endpoint.prices
    return prices


def _pinned_tag(spec: RunSpec) -> str | None:
    return spec.providers[0] if spec.providers else None


def _as_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def write_probe_run(  # noqa: PLR0913 (one keyword per part of the record it writes)
    runs_dir: Path,
    trace_name: str,
    conversation: str,
    run: ProbeRun,
    specs: Sequence[RunSpec],
    *,
    endpoints: Mapping[str, EndpointSnapshot] | None = None,
    prices: Mapping[str, SpecPrices] | None = None,
    notes: Sequence[str] = (),
    sweep: SweepInfo | None = None,
    precheck: Sequence[ProbeResult] = (),
    trace_prompt_tokens: int | None = None,
    listing_prices: Mapping[str, Prices] | None = None,
    reported_average: ReportedAverage | None = None,
) -> Path:
    """Write one `<slug>.jsonl` per spec plus the run's `run.json`.

    `precheck`, when the run planned an availability check, is written to its own
    `precheck.jsonl` -- never merged into a spec's file, so `load_probe_run` never hands it
    back as part of a spec's own records.
    """
    created = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / trace_name / created
    run_dir.mkdir(parents=True, exist_ok=True)

    refs: list[RunRef] = []
    for spec in specs:
        file_name = f"{spec.slug}.jsonl"
        with (run_dir / file_name).open("w", encoding="utf-8") as fh:
            for record in run.records[spec.label]:
                fh.write(record.model_dump_json() + "\n")
        refs.append(
            RunRef(
                label=spec.label,
                slug=spec.slug,
                target=spec.target.name,
                model=spec.model,
                providers=spec.providers,
                file=file_name,
                kind=spec.target.kind,
            )
        )

    if precheck:
        with (run_dir / _PRECHECK_FILE).open("w", encoding="utf-8") as fh:
            for record in precheck:
                fh.write(record.model_dump_json() + "\n")
        precheck_summary = PrecheckSummary(
            requests=len(precheck),
            spend_usd=precheck_spend(precheck, prices or {}),
            worst_case_usd=precheck_worst_case(precheck, prices or {}),
        )
    else:
        precheck_summary = None

    meta = ProbeRunMeta(
        trace=trace_name,
        conversation=conversation,
        created=created,
        run_hex=run.run_hex,
        options=run.options,
        specs=refs,
        ttl_s=run.options.ttl_s,
        endpoints=dict(endpoints or {}),
        prices=dict(prices or {}),
        notes=list(notes),
        sweep=sweep,
        precheck=precheck_summary,
        trace_prompt_tokens=trace_prompt_tokens,
        listing_prices=None if listing_prices is None else dict(listing_prices),
        reported_average=reported_average,
    )
    (run_dir / "run.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run_dir


def load_probe_run(run_dir: Path) -> tuple[ProbeRunMeta, dict[str, list[ProbeResult]]]:
    """Read back a probe run: its meta, and each spec's records.

    A run's `precheck.jsonl`, when it wrote one, is not among these: it is not a spec's own
    record and would flatter or skew nothing correctly if it were folded in.
    """
    meta = ProbeRunMeta.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    records: dict[str, list[ProbeResult]] = {}
    for ref in meta.specs:
        lines = (run_dir / ref.file).read_text(encoding="utf-8").splitlines()
        records[ref.label] = [
            ProbeResult.model_validate_json(line) for line in lines if line.strip()
        ]
    return meta, records


def read_protocol(run_dir: Path) -> str:
    """A persisted run's protocol; `full` for a run written before the field existed."""
    raw = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        protocol = cast("dict[str, object]", raw).get("protocol")
        if isinstance(protocol, str):
            return protocol
    return "full"
