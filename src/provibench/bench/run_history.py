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
stamp, the trace, the protocol, and the endpoint facts the run's snapshot recorded.

A full replay records no per-spec price and no streamed throughput request, so its rows
read `-` where a probe's carry a listed price, a TTFT and a tok/s; the run still lists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Set
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
    """The listed input price at run time, from the run's `prices` block."""
    listed_cache_read: float | None = None
    listed_source: str = NO_PRICE_SOURCE
    spend_usd: float | None = None
    """What the spec cost (see `bench.spend.spec_spend`), recomputed from the run's own
    records, so an older run has one too; `None` when the run recorded no price to compute
    it from, and always `None` for a full replay."""
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
    """One spec, trace and protocol's runs for a distinct sparkline."""

    spec: str
    provider: str
    trace: str
    protocol: str
    runs: int
    hit_rates: list[float | None] = Field(default_factory=list[float | None])
    """One entry per run of the spec, in date order; `None` where a run measured none."""


def created_at(created: str) -> float | None:
    """A run's `created` stamp as seconds since the epoch, or `None` when it is not one.

    The stamp is written by the run itself and is UTC (`%Y%m%d-%H%M%S`); a directory whose
    stamp was edited by hand, or a format from an older version, still reads as a run — it
    just cannot be aged, so `--since` keeps it. It sorts as the oldest run, so it is
    leftmost in its sparkline.
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
    extra_labels: Set[str] = frozenset(),
) -> list[RunNumbers]:
    """Every run under `runs_dir` the filters keep, oldest first.

    A run is kept when the model filter selects at least one of its specs — the other
    specs are dropped from it, because `history MODEL` is about that model — and when
    its trace is the one asked for. Age filtering is separate so the command can tell
    "no run was ever recorded" from "no run this recent".

    `extra_labels` keeps a row by its exact `spec` label regardless of `matches_model`: a
    native spec's `model` is the name its own target serves it under (`deepseek-flash`),
    not the OpenRouter slug `history` was asked about, so the command resolves that slug
    through the configured targets' aliases (the same aliases `sweep` reads) and passes the
    native labels it maps to here — `bench` itself never reads `targets.toml`.
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
                        or row.spec in extra_labels
                    ]
                }
            )
            for run in kept
        ]
        kept = [run for run in kept if run.rows]
    if trace is not None:
        kept = [run for run in kept if run.trace == trace]
    return sorted(kept, key=_run_order)


def keep_fresh(
    runs: Sequence[RunNumbers], *, since_s: int | None = None, now: float
) -> list[RunNumbers]:
    """The runs not older than `since_s`, in the order given.

    Separate from `scan_runs` so a caller can tell "no run was ever recorded" from "no run
    this recent" — only the first is a `not_found` — without reading the run directories
    twice. A run whose stamp cannot be parsed is kept: it cannot be shown to be old.
    """
    if since_s is None:
        return list(runs)
    cutoff = now - since_s
    return [run for run in runs if _fresh(run, cutoff)]


def rows_in_order(runs: Iterable[RunNumbers]) -> list[RunRow]:
    """Every row of every run, for a table read one provider at a time over time."""
    return sorted(
        (row for run in runs for row in run.rows),
        key=lambda row: (row.provider, created_at(row.created) or 0.0, row.spec),
    )


def history_series(runs: Sequence[RunNumbers]) -> list[Series]:
    """One series per spec, trace and protocol, in table order.

    Runs of one spec in date order, with a `None` where a run measured no hit rate, so the
    sparkline can preserve the missing run without the run count going missing.
    """
    grouped: dict[tuple[str, str, str], list[RunRow]] = {}
    for row in rows_in_order(runs):
        grouped.setdefault((row.spec, row.trace, row.protocol), []).append(row)
    return [
        Series(
            spec=key[0],
            provider=rows[0].provider,
            trace=key[1],
            protocol=key[2],
            runs=len(rows),
            hit_rates=[row.hit_rate for row in rows],
        )
        for key, rows in grouped.items()
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
                ref.label,
                records[ref.label],
                prices=meta.prices.get(ref.label),
                unpinned_gateway=ref.kind == "openrouter" and not ref.providers,
                trace_prompt_tokens=meta.trace_prompt_tokens,
                listing=meta.listing_prices,
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
    # eff_per_m_prompt is read off the summary `report` prints, so the two cannot differ;
    # listed_input/listed_source are the summary's own listed-price fields, never a fitted
    # reprice, so two runs of an unpinned spec cannot show a price change that is fit noise
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
        listed_input=summary.listed_input,
        listed_cache_read=summary.listed_cache_read,
        listed_source=summary.listed_source,
        spend_usd=summary.spend_usd,
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
