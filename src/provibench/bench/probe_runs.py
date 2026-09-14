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

from provibench.bench.estimate import SpecPrices, match_endpoint
from provibench.bench.openrouter import Endpoint
from provibench.bench.probe_models import ProbeOptions, ProbeResult, ProbeRun
from provibench.bench.replay import RunRef
from provibench.bench.selection import SweepInfo
from provibench.bench.targets import RunSpec

__all__ = [
    "EndpointSnapshot",
    "ProbeRunMeta",
    "SweepInfo",
    "endpoint_snapshot",
    "load_probe_run",
    "read_protocol",
    "write_probe_run",
]
"""`SweepInfo` lives in `bench.selection` with the criteria it records; it is re-exported
here because a run directory is where a reader meets it."""


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
) -> Path:
    """Write one `<slug>.jsonl` per spec plus the run's `run.json`."""
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
    )
    (run_dir / "run.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run_dir


def load_probe_run(run_dir: Path) -> tuple[ProbeRunMeta, dict[str, list[ProbeResult]]]:
    """Read back a probe run: its meta, and each spec's records."""
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
