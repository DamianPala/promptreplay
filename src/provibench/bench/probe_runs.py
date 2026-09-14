"""Persisting and reading probe runs: `run.json` plus one `<slug>.jsonl` per spec.

A run directory is the record of what was spent, so it carries everything needed to
re-read the result offline: the rungs and repeats that ran, the endpoint snapshot the
prices came from, and the price of each spec at run time. `report` never needs a network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from provibench.bench.estimate import SpecPrices
from provibench.bench.openrouter import Endpoint
from provibench.bench.probe_models import ProbeOptions, ProbeResult, ProbeRun
from provibench.bench.replay import RunRef
from provibench.bench.targets import RunSpec


class ProbeRunMeta(BaseModel):
    """The `run.json` of a persisted probe run."""

    protocol: Literal["probe"] = "probe"
    trace: str
    conversation: str
    created: str
    run_hex: str
    options: ProbeOptions
    specs: list[RunRef]
    endpoints: dict[str, list[Endpoint]] = Field(default_factory=dict)
    prices: dict[str, SpecPrices] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def write_probe_run(
    runs_dir: Path,
    trace_name: str,
    conversation: str,
    run: ProbeRun,
    specs: Sequence[RunSpec],
    *,
    endpoints: Mapping[str, list[Endpoint]] | None = None,
    prices: Mapping[str, SpecPrices] | None = None,
    notes: Sequence[str] = (),
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
            )
        )

    meta = ProbeRunMeta(
        trace=trace_name,
        conversation=conversation,
        created=created,
        run_hex=run.run_hex,
        options=run.options,
        specs=refs,
        endpoints=dict(endpoints or {}),
        prices=dict(prices or {}),
        notes=list(notes),
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
