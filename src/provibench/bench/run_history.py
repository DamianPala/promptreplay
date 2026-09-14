"""A runs directory read as a time series: what each run measured, and when.

Providers change routing, quantization, cache config and prices from week to week, so the
realistic use of this tool is a re-run every few days rather than one benchmark. A run
directory already holds everything such a comparison needs; this module is what turns a
directory of them into rows a reader can line up — `history` lists them, and
`bench.run_delta` subtracts two of them.

Every run goes back through the loader that owns its protocol (`load_probe_run` for a
probe, `load_run` for a full replay) and then through the same summarising the report
does, so a number printed here cannot disagree with the number `report` prints for that
run. What this module adds is the run-level context the reports leave implicit: the date
stamp, the trace, and the endpoint snapshot the run recorded.

A full replay carries no endpoint snapshot and no streamed throughput request, so its rows
read `-` where a probe's carry a listed price, a TTFT and a tok/s; the run still lists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from provibench.bench.probe_runs import EndpointSnapshot, load_probe_run, read_protocol
from provibench.bench.probe_summary import ProbeSummary, summarize_probe
from provibench.bench.replay import RunRef, load_run
from provibench.bench.summary import summarize

__all__ = [
    "RunNumbers",
    "RunRow",
    "Series",
    "created_at",
    "history_series",
    "keep_fresh",
    "matches_model",
    "models_in",
    "read_run",
    "rows_in_order",
    "scan_runs",
]

_STAMP_FORMAT = "%Y%m%d-%H%M%S"
NO_PRICE_SOURCE = "n/a"
"""Where a row's listed price came from: `n/a` means the run recorded none."""


class RunRow(BaseModel):
    """One spec in one run: the numbers a history line is made of.

    The run-level fields are repeated on every row because a row is what a table prints
    and what `--json` carries; the label, target, model and provider are the spec's, and
    the rest is what the run measured or the listing said about the endpoint at the time.
    """

    trace: str
    created: str
    run_dir: Path
    protocol: str
    spec: str
    target: str
    model: str
    provider: str
    """The pinned tag, or the target's name for a spec that pins nothing."""
    hit_rate: float | None = None
    """Warm reads that found the cache, as a fraction; `None` when none was served."""
    eff_per_m_prompt: float | None = None
    ttft_ms: float | None = None
    gen_tok_s: float | None = None
    errors: int = 0
    listed_input: float | None = None
    """The endpoint's listed input price at run time, from the run's snapshot."""
    listed_cache_read: float | None = None
    listed_source: str = NO_PRICE_SOURCE
    quantization: str | None = None
    context_length: int | None = None
    uptime_1d: float | None = None
    status: str | None = None


class RunNumbers(BaseModel):
    """One run directory: when it ran, under which trace and protocol, and its rows."""

    run_dir: Path
    trace: str
    created: str
    protocol: str
    rows: list[RunRow] = Field(default_factory=list[RunRow])


class Series(BaseModel):
    """One spec's runs: the hit rate of each, oldest first, for the sparkline block."""

    spec: str
    provider: str
    runs: int
    hit_rates: list[float | None] = Field(default_factory=list[float | None])
    """One entry per run of the spec, in date order; `None` where a run measured none."""


def created_at(created: str) -> float | None:
    """A run's `created` stamp as seconds since the epoch, or `None` when it is not one.

    The stamp is written by the run itself and is UTC (`%Y%m%d-%H%M%S`); a directory whose
    stamp was edited by hand, or a format from an older version, still reads as a run — it
    just cannot be aged, so `--since` keeps it.
    """
    try:
        return datetime.strptime(created, _STAMP_FORMAT).replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def matches_model(model: str, *, carried: str, target: str) -> bool:
    """Whether a spec is one MODEL names: its own model, or its whole `target:model` head.

    A gateway spec carries the OpenRouter slug, so `history deepseek/deepseek-v4.1-flash`
    finds it; a native spec carries the name its own endpoint serves (`deepseek-flash`),
    which is the only name it can be asked about. The head form is taken too, because that
    is how the label reads in a report — `or:deepseek/deepseek-v4.1-flash`.
    """
    return model in (carried, f"{target}:{carried}")


def read_run(run_dir: Path) -> RunNumbers:
    """One run directory, whatever protocol wrote it, through the loader that owns it."""
    if read_protocol(run_dir) == "probe":
        return _probe_run(run_dir)
    return _full_run(run_dir)


def scan_runs(
    runs_dir: Path,
    *,
    model: str | None = None,
    trace: str | None = None,
    since_s: int | None = None,
    now: float | None = None,
) -> list[RunNumbers]:
    """Every run under `runs_dir` the filters keep, oldest first.

    A run is kept when the model filter selects at least one of its specs — the other
    specs are dropped from it, because `history MODEL` is about that model — when its
    trace is the one asked for, and when it is not older than `--since`.
    """
    kept = [run for run in map(read_run, _run_dirs(runs_dir)) if run.rows]
    if model is not None:
        kept = [
            run.model_copy(
                update={
                    "rows": [
                        row
                        for row in run.rows
                        if matches_model(model, carried=row.model, target=row.target)
                    ]
                }
            )
            for run in kept
        ]
        kept = [run for run in kept if run.rows]
    if trace is not None:
        kept = [run for run in kept if run.trace == trace]
    return sorted(keep_fresh(kept, since_s=since_s, now=now), key=_run_order)


def keep_fresh(
    runs: Sequence[RunNumbers], *, since_s: int | None = None, now: float | None = None
) -> list[RunNumbers]:
    """The runs not older than `since_s`, in the order given.

    Separate from `scan_runs` so a caller can tell "no run was ever recorded" from "no run
    this recent" — only the first is a `not_found` — without reading the run directories
    twice. A run whose stamp cannot be parsed is kept: it cannot be shown to be old.
    """
    if since_s is None:
        return list(runs)
    cutoff = (0.0 if now is None else now) - since_s
    return [run for run in runs if _fresh(run, cutoff)]


def rows_in_order(runs: Iterable[RunNumbers]) -> list[RunRow]:
    """Every row of every run, for a table read one provider at a time over time."""
    return sorted(
        (row for run in runs for row in run.rows),
        key=lambda row: (row.provider, created_at(row.created) or 0.0, row.spec),
    )


def history_series(runs: Sequence[RunNumbers]) -> list[Series]:
    """One series per spec, in the order `rows_in_order` prints those specs.

    Runs of one spec in date order, with a `None` where a run measured no hit rate, so the
    sparkline can skip it without the run count going missing.
    """
    grouped: dict[str, list[RunRow]] = {}
    for row in rows_in_order(runs):
        grouped.setdefault(row.spec, []).append(row)
    return [
        Series(
            spec=spec,
            provider=rows[0].provider,
            runs=len(rows),
            hit_rates=[row.hit_rate for row in rows],
        )
        for spec, rows in grouped.items()
    ]


def models_in(runs: Sequence[RunNumbers]) -> list[str]:
    """Every model the runs carry, without repeats, in the table's reading order."""
    found: list[str] = []
    for row in rows_in_order(runs):
        if row.model not in found:
            found.append(row.model)
    return found


def _probe_run(run_dir: Path) -> RunNumbers:
    """A probe run read through `load_probe_run` and summarised the way `report` summarises it."""
    meta, records = load_probe_run(run_dir)
    rows = [
        _probe_row(
            ref=ref,
            run_dir=run_dir,
            trace=meta.trace,
            created=meta.created,
            summary=summarize_probe(
                ref.label, records[ref.label], prices=meta.prices.get(ref.label)
            ),
            snapshot=meta.endpoints.get(ref.label),
        )
        for ref in meta.specs
    ]
    return RunNumbers(
        run_dir=run_dir, trace=meta.trace, created=meta.created, protocol="probe", rows=rows
    )


def _probe_row(
    *,
    ref: RunRef,
    run_dir: Path,
    trace: str,
    created: str,
    summary: ProbeSummary,
    snapshot: EndpointSnapshot | None,
) -> RunRow:
    return RunRow(
        trace=trace,
        created=created,
        run_dir=run_dir,
        protocol="probe",
        spec=ref.label,
        target=ref.target,
        model=ref.model,
        provider=_provider(ref),
        hit_rate=summary.hit_rate,
        eff_per_m_prompt=summary.eff_per_m_prompt,
        ttft_ms=summary.ttft_ms,
        gen_tok_s=summary.gen_tok_s,
        errors=summary.errors,
        listed_input=None if snapshot is None else snapshot.price_input,
        listed_cache_read=None if snapshot is None else snapshot.price_cache_read,
        listed_source=NO_PRICE_SOURCE if snapshot is None else snapshot.source,
        quantization=None if snapshot is None else snapshot.quantization,
        context_length=None if snapshot is None else snapshot.context_length,
        uptime_1d=None if snapshot is None else snapshot.uptime_1d,
        status=None if snapshot is None or snapshot.status is None else str(snapshot.status),
    )


def _full_run(run_dir: Path) -> RunNumbers:
    """A full replay read through `load_run`: totals per spec, no snapshot and no stream."""
    meta, results = load_run(run_dir)
    rows: list[RunRow] = []
    for ref in meta.runs:
        summary = summarize(ref.label, results[ref.label], ref.providers)
        rows.append(
            RunRow(
                trace=meta.trace,
                created=meta.created,
                run_dir=run_dir,
                protocol=meta.protocol,
                spec=ref.label,
                target=ref.target,
                model=ref.model,
                provider=_provider(ref),
                hit_rate=summary.hit_ratio,
                eff_per_m_prompt=summary.effective_per_m_prompt,
                errors=summary.errors,
            )
        )
    return RunNumbers(
        run_dir=run_dir, trace=meta.trace, created=meta.created, protocol=meta.protocol, rows=rows
    )


def _provider(ref: RunRef) -> str:
    """The endpoint a spec is read by: its pinned tag, else the target's own name."""
    return ref.providers[0] if ref.providers else ref.target


def _fresh(run: RunNumbers, cutoff: float) -> bool:
    at = created_at(run.created)
    return at is None or at >= cutoff


def _run_order(run: RunNumbers) -> tuple[float, str]:
    return (created_at(run.created) or 0.0, str(run.run_dir))


def _run_dirs(runs_dir: Path) -> list[Path]:
    """Every `<trace>/<stamp>/` directory under the runs dir that holds a `run.json`."""
    return [
        path for path in runs_dir.glob("*/*") if path.is_dir() and (path / "run.json").is_file()
    ]
