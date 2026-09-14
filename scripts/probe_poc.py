"""Probe PoC: sample a provider's prompt-cache behaviour from a few recorded turns.

Instead of replaying a whole conversation, this walks a handful of 1-based turn
indices ("rungs") of the trace's main conversation. For each rung it sends turn `k`
once ("cold") and turn `k+1` a few times ("warm"), so the cold request's prompt is a
cacheable prefix of the warm request. Each rung carries its own nonce
(`provibench-probe:<run_hex>:<k>`) prepended to the first system block: every cold
request is then truly cold, because rung 13 and rung 30 share the same opening bytes
with rung 1 and would otherwise read back rung 1's cache. A rung whose cold request
fails is skipped: there is nothing cached to warm, so it does not score.

    uv run python scripts/probe_poc.py TRACE SPEC [SPEC ...] \
        [--rungs 1,13,30] [--repeats 5,2,2] [--gap 1.0] [--timeout 300] \
        [--targets PATH] [--output-file FILE.json]

Exit code 0 when every request got a 2xx, 1 otherwise (the JSON output is written
either way); 2 for invalid input, a missing API key, or an impossible rung. A partial
run (interrupt or unexpected failure) still writes the records collected so far.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel

import provibench

# The private replay helpers are reused deliberately: this PoC rides the same
# request path as `replay` so its measurements are comparable.
from provibench.bench.replay import (
    ReplayOptions,
    ReplayResult,
    _build_headers,  # pyright: ignore[reportPrivateUsage]
    _enrich_openrouter,  # pyright: ignore[reportPrivateUsage]
    _parse_body,  # pyright: ignore[reportPrivateUsage]
    _post,  # pyright: ignore[reportPrivateUsage]
    prepare_body,
)
from provibench.bench.targets import RunSpec, load_targets, parse_run_spec, resolve_api_key
from provibench.bench.trace import TraceEntry, load_trace, main_conversation

type Role = Literal["cold", "warm"]

_NONCE_PREFIX = "provibench-probe:"
_TWOXX_MIN = 200
_TWOXX_MAX = 300
_RETRY_STATUSES = (429, 503)
_RETRY_DELAYS = (2.0, 5.0)


# --- nonce and rung selection -------------------------------------------------


def new_run_hex() -> str:
    """The run's shared hex; every rung derives its own nonce from it."""
    return secrets.token_hex(6)


def rung_nonce(run_hex: str, rung: int) -> str:
    """The nonce for one rung, identical across every spec of the run."""
    return f"{_NONCE_PREFIX}{run_hex}:{rung}"


def inject_nonce(body: dict[str, Any], nonce: str) -> dict[str, Any]:
    """A copy of `body` with the nonce prepended to its first system block.

    The block-level cache keys on the exact block text, so stamping a fresh nonce
    means a cold request can only be read back by this run's own warm requests.
    `cache_control` and every other field are left untouched. A body the nonce cannot
    be stamped into is an error rather than a silently unisolated request.
    """
    out = copy.deepcopy(body)
    system = out.get("system")
    if system is None:
        raise ValueError("body has no 'system' prompt to stamp the nonce into")
    if isinstance(system, str):
        out["system"] = f"{nonce}\n{system}"
        return out
    if not isinstance(system, list) or not system:
        kind = type(cast("object", system)).__name__
        raise ValueError(
            f"body['system'] must be a string or a non-empty list of blocks, got {kind}"
        )
    blocks = cast("list[Any]", system)
    # `hasattr` rather than `isinstance(dict)` keeps the element typed `Any`, so the
    # block's own `cache_control` and siblings are never re-typed or reordered.
    if not hasattr(blocks[0], "get") or not isinstance(blocks[0].get("text"), str):
        raise ValueError("body['system'][0] has no 'text' block to stamp the nonce into")
    blocks[0]["text"] = f"{nonce}\n{blocks[0]['text']}"
    return out


def build_probe_body(
    entry: TraceEntry, spec: RunSpec, opts: ReplayOptions, nonce: str
) -> dict[str, Any]:
    """The recorded body prepared for replay, then stamped with the run nonce."""
    return inject_nonce(prepare_body(entry.body, spec, opts), nonce)


def parse_int_list(raw: str) -> list[int]:
    """A non-empty comma-separated list of integers."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"expected a comma-separated list of integers, got {raw!r}")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"expected a comma-separated list of integers, got {raw!r}") from exc


def broadcast_repeats(rungs: Sequence[int], repeats: Sequence[int]) -> list[int]:
    """One repeat count per rung; a short `repeats` list repeats its last value."""
    if not repeats:
        raise ValueError("--repeats must contain at least one value")
    return [repeats[min(index, len(repeats) - 1)] for index in range(len(rungs))]


def validate_rungs(rungs: Sequence[int], conversation: Sequence[TraceEntry]) -> None:
    """Every rung must point at a turn `k` that has a following turn `k+1`."""
    if not conversation:
        raise ValueError("this trace has no conversations")
    total = len(conversation)
    for rung in rungs:
        if rung < 1:
            raise ValueError(f"rung must be at least 1, got {rung}")
        if rung + 1 > total:
            raise ValueError(
                f"rung {rung} needs turn {rung + 1}, but the conversation has {total} turn(s)"
            )


# --- records ------------------------------------------------------------------


class RequestRecord(BaseModel):
    """One HTTP request: the cold request of a rung, or one of its warm attempts."""

    spec: str
    rung: int
    nonce: str
    role: Role
    attempt: int
    status: int
    latency_ms: float
    prompt_total: int
    input_tokens: int
    cached: int
    cache_write: int
    provider: str | None = None
    message_id: str | None = None
    native_tokens_cached: int | None = None
    cached_source: Literal["response", "openrouter-generation"] = "response"
    retries: int = 0
    note: str | None = None
    error: str | None = None


class RungAggregate(BaseModel):
    """A rung's cold/warm requests folded into the numbers the report shows."""

    spec: str
    rung: int
    nonce: str
    prompt_cold: int
    prefix: int
    cached_cold: int
    hit_rate: float
    frac: float | None
    hits: list[float | Literal["x"]]
    ttft_cold_ms: float
    ttft_warm_ms: float | None
    cache_write_cold: int
    skipped: bool
    cold_error: str | None = None
    errors: int


class ProbeRun(BaseModel):
    """The `--output-file` document."""

    run_hex: str
    trace: str
    started: str
    requests: list[RequestRecord]
    rungs: list[RungAggregate]


# --- request plumbing ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Call:
    rung: int
    role: Role
    attempt: int
    nonce: str


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """Everything the run needs that comes from the command line."""

    trace: Path
    specs: Sequence[RunSpec]
    rungs: Sequence[int]
    repeats: Sequence[int]
    gap_s: float = 1.0
    timeout_s: float = 300.0

    def options(self) -> ReplayOptions:
        return ReplayOptions(max_tokens=1, timeout_s=self.timeout_s)


@dataclass(slots=True)
class _Probe:
    """The per-spec context shared by its requests."""

    spec: RunSpec
    opts: ReplayOptions
    api_key: str
    client: httpx.AsyncClient
    progress: Callable[[RequestRecord], None] | None = None


@dataclass(slots=True)
class _Outcome:
    record: RequestRecord
    replay: ReplayResult | None


@dataclass(slots=True)
class ProbeState:
    """Mutable run state shared with the caller.

    Outcomes are appended as each response lands, so a run that is interrupted or
    fails partway still has every record that was paid for.
    """

    run_hex: str
    trace: str
    started: str
    outcomes: list[_Outcome] = field(default_factory=list[_Outcome])

    @classmethod
    def start(cls, trace: Path) -> ProbeState:
        return cls(run_hex=new_run_hex(), trace=str(trace), started=datetime.now(UTC).isoformat())

    def build(self, config: ProbeConfig) -> ProbeRun:
        return ProbeRun(
            run_hex=self.run_hex,
            trace=self.trace,
            started=self.started,
            requests=[outcome.record for outcome in self.outcomes],
            rungs=[
                summarize_rung(
                    spec.label,
                    rung,
                    rung_nonce(self.run_hex, rung),
                    _records_for(self.outcomes, spec.label, rung),
                )
                for spec in config.specs
                for rung in config.rungs
            ],
        )


def _report(probe: _Probe, record: RequestRecord) -> None:
    if probe.progress is not None:
        probe.progress(record)


def _error_record(
    probe: _Probe, call: _Call, started: float, retries: int, error: str
) -> RequestRecord:
    return RequestRecord(
        spec=probe.spec.label,
        rung=call.rung,
        nonce=call.nonce,
        role=call.role,
        attempt=call.attempt,
        status=0,
        latency_ms=(perf_counter() - started) * 1000,
        prompt_total=0,
        input_tokens=0,
        cached=0,
        cache_write=0,
        retries=retries,
        error=error,
    )


def _outcome_from(
    probe: _Probe,
    entry: TraceEntry,
    resp: httpx.Response,
    call: _Call,
    *,
    latency_ms: float,
    retries: int,
) -> _Outcome:
    """The record (and replay result) for one completed HTTP response."""
    parsed = _parse_body(resp)
    usage = parsed.usage
    record = RequestRecord(
        spec=probe.spec.label,
        rung=call.rung,
        nonce=call.nonce,
        role=call.role,
        attempt=call.attempt,
        status=resp.status_code,
        latency_ms=latency_ms,
        prompt_total=usage.prompt_total,
        input_tokens=usage.input_tokens,
        cached=usage.cache_read_input_tokens,
        cache_write=usage.cache_creation_input_tokens,
        provider=parsed.provider,
        message_id=parsed.message_id,
        retries=retries,
        error=parsed.error,
    )
    replay = ReplayResult(
        seq=entry.seq,
        turn=call.rung,
        status=resp.status_code,
        latency_ms=latency_ms,
        message_id=parsed.message_id,
        model=parsed.model,
        provider=parsed.provider,
        requested_providers=probe.spec.providers,
        usage=usage,
        prompt_total=usage.prompt_total,
        cached=usage.cache_read_input_tokens,
        cache_write=usage.cache_creation_input_tokens,
        output_tokens=usage.output_tokens,
        error=parsed.error,
    )
    return _Outcome(record, replay)


async def _send(
    probe: _Probe, state: ProbeState, entry: TraceEntry, body: dict[str, Any], call: _Call
) -> _Outcome:
    """POST one prepared body, retrying transient overloads, and record the outcome.

    A 429 or 503 is retried with the backoff schedule in `_RETRY_DELAYS`; the bytes are
    byte-identical across attempts, so the replay is safe. Only the last attempt's wall
    clock is recorded, so a backoff wait never inflates latency.
    """
    headers = _build_headers(entry, probe.api_key)
    retries = 0
    while True:
        started = perf_counter()
        try:
            resp = await _post(probe.client, probe.spec, headers, body, probe.opts)
        except httpx.HTTPError as exc:
            outcome = _Outcome(_error_record(probe, call, started, retries, str(exc)), None)
            break
        if resp.status_code in _RETRY_STATUSES and retries < len(_RETRY_DELAYS):
            await asyncio.sleep(_RETRY_DELAYS[retries])
            retries += 1
            continue
        outcome = _outcome_from(
            probe, entry, resp, call, latency_ms=(perf_counter() - started) * 1000, retries=retries
        )
        break
    state.outcomes.append(outcome)
    _report(probe, outcome.record)
    return outcome


def _native_cached(generation: dict[str, Any]) -> int | None:
    value = generation.get("native_tokens_cached")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _apply_enrichment(outcome: _Outcome) -> None:
    """Copy the OpenRouter generation lookup's provider/native counts onto the record.

    Response usage stays in `cached`/`prompt_total`; the native count is kept alongside
    it so a divergence between the two is visible. When the response reports no cached
    tokens at all but the gateway's native count has some, the native value is used and
    `cached_source` says so.
    """
    replay = outcome.replay
    if replay is None:
        return
    if replay.note:
        outcome.record.note = replay.note
    if replay.generation is None:
        return
    outcome.record.provider = replay.provider
    native = _native_cached(replay.generation)
    outcome.record.native_tokens_cached = native
    if outcome.record.cached == 0 and native is not None and native > 0:
        outcome.record.cached = native
        outcome.record.cached_source = "openrouter-generation"


async def _enrich_spec(probe: _Probe, outcomes: list[_Outcome]) -> None:
    """Run the existing generation enrichment for openrouter targets, best-effort."""
    if probe.spec.target.kind != "openrouter":
        return
    replays = [outcome.replay for outcome in outcomes if outcome.replay is not None]
    try:
        await _enrich_openrouter(probe.spec, replays, probe.client, probe.api_key)
    except (httpx.HTTPError, ValueError) as exc:
        message = f"generation enrichment failed: {exc}"
        for outcome in outcomes:
            outcome.record.note = message
        return
    for outcome in outcomes:
        _apply_enrichment(outcome)


async def _probe_spec(
    probe: _Probe, state: ProbeState, config: ProbeConfig, conversation: Sequence[TraceEntry]
) -> None:
    """Send every rung of one spec; a rung whose cold request fails is skipped."""
    counts = broadcast_repeats(config.rungs, config.repeats)
    for rung, count in zip(config.rungs, counts, strict=True):
        nonce = rung_nonce(state.run_hex, rung)
        cold_entry = conversation[rung - 1]
        cold_body = build_probe_body(cold_entry, probe.spec, probe.opts, nonce)
        cold = await _send(probe, state, cold_entry, cold_body, _Call(rung, "cold", 0, nonce))
        if not _served(cold.record):
            continue  # nothing was cached, so there is nothing to warm
        warm_entry = conversation[rung]
        warm_body = build_probe_body(warm_entry, probe.spec, probe.opts, nonce)
        for attempt in range(1, count + 1):
            # A cache write can land after the response returns, so wait before every
            # warm attempt, the first one included.
            if config.gap_s > 0:
                await asyncio.sleep(config.gap_s)
            await _send(probe, state, warm_entry, warm_body, _Call(rung, "warm", attempt, nonce))


# --- aggregation --------------------------------------------------------------


def _is_twoxx(status: int) -> bool:
    return _TWOXX_MIN <= status < _TWOXX_MAX


def _served(record: RequestRecord) -> bool:
    return _is_twoxx(record.status) and record.error is None


def _failed(record: RequestRecord) -> bool:
    return record.error is not None or not _is_twoxx(record.status)


def _fraction(cached: int, prefix: int) -> float:
    if prefix <= 0:
        return 0.0
    return min(cached / prefix, 1.0)


def _hit(record: RequestRecord, prefix: int) -> float | Literal["x"]:
    """One warm attempt's cache fraction, or "x" when the attempt failed."""
    if not _served(record):
        return "x"
    return round(_fraction(record.cached, prefix), 2)


def summarize_rung(
    spec: str, rung: int, nonce: str, records: Sequence[RequestRecord]
) -> RungAggregate:
    """Fold one rung's cold + warm records into hit rate, fractions and timings.

    A rung whose cold request was not served is `skipped`: it scores nothing (no frac,
    an empty `hits`) and carries the cold error for the report.
    """
    cold = next((record for record in records if record.role == "cold"), None)
    warm = [record for record in records if record.role == "warm"]
    skipped = cold is None or not _served(cold)
    prefix = cold.prompt_total if cold is not None else 0
    served = [record for record in warm if _served(record)]
    hits = [record for record in served if record.cached > 0]
    warm_latencies = [record.latency_ms for record in served]
    return RungAggregate(
        spec=spec,
        rung=rung,
        nonce=nonce,
        prompt_cold=prefix,
        prefix=prefix,
        cached_cold=cold.cached if cold is not None else 0,
        hit_rate=len(hits) / len(served) if served else 0.0,
        frac=fmean(_fraction(record.cached, prefix) for record in hits) if hits else None,
        hits=[_hit(record, prefix) for record in warm],
        # Non-streaming responses give no time-to-first-token; the full-response wall
        # clock is the closest available proxy for both columns.
        ttft_cold_ms=cold.latency_ms if cold is not None else 0.0,
        ttft_warm_ms=median(warm_latencies) if warm_latencies else None,
        cache_write_cold=cold.cache_write if cold is not None else 0,
        skipped=skipped,
        cold_error=cold.error if cold is not None else None,
        errors=sum(1 for record in records if _failed(record)),
    )


def _records_for(outcomes: Sequence[_Outcome], spec: str, rung: int) -> list[RequestRecord]:
    return [
        outcome.record
        for outcome in outcomes
        if outcome.record.spec == spec and outcome.record.rung == rung
    ]


async def run_probe(
    config: ProbeConfig,
    entries: Sequence[TraceEntry],
    api_keys: Mapping[str, str],
    *,
    state: ProbeState,
    client: httpx.AsyncClient,
    progress: Callable[[RequestRecord], None] | None = None,
) -> ProbeRun:
    """Send every rung of every spec, specs sequentially, rungs in order."""
    conversation = sorted(main_conversation(list(entries)), key=lambda entry: entry.seq)
    validate_rungs(config.rungs, conversation)

    for spec in config.specs:
        probe = _Probe(
            spec=spec,
            opts=config.options(),
            api_key=api_keys[spec.label],
            client=client,
            progress=progress,
        )
        start = len(state.outcomes)
        await _probe_spec(probe, state, config, conversation)
        await _enrich_spec(probe, state.outcomes[start:])

    return state.build(config)


def exit_code(records: Sequence[RequestRecord]) -> int:
    """0 when every request got a 2xx, 1 otherwise."""
    return 0 if all(not _failed(record) for record in records) else 1


# --- reporting ----------------------------------------------------------------

_HEADERS = (
    "spec",
    "rung",
    "prompt",
    "cached cold",
    "hit_rate",
    "frac",
    "hits",
    "cold ms",
    "warm ms",
    "errors",
)


def _format_hits(values: Sequence[float | Literal["x"]]) -> str:
    return (
        "[" + ", ".join("x" if isinstance(value, str) else f"{value:.2f}" for value in values) + "]"
    )


def _format_row(aggregate: RungAggregate) -> list[str]:
    return [
        aggregate.spec,
        str(aggregate.rung),
        str(aggregate.prompt_cold),
        str(aggregate.cached_cold),
        f"{aggregate.hit_rate:.2f}",
        "-" if aggregate.frac is None else f"{aggregate.frac:.2f}",
        _format_hits(aggregate.hits),
        f"{aggregate.ttft_cold_ms:.0f}",
        "-" if aggregate.ttft_warm_ms is None else f"{aggregate.ttft_warm_ms:.0f}",
        str(aggregate.errors),
    ]


def _join(cells: Sequence[str], widths: Sequence[int]) -> str:
    return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def _widths(rows: Sequence[Sequence[str]]) -> list[int]:
    return [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(_HEADERS)
    ]


def _spec_summaries(rungs: Sequence[RungAggregate]) -> list[str]:
    order: list[str] = []
    for aggregate in rungs:
        if aggregate.spec not in order:
            order.append(aggregate.spec)
    lines: list[str] = []
    for spec in order:
        fracs = [
            aggregate.frac if aggregate.frac is not None else 0.0
            for aggregate in rungs
            if aggregate.spec == spec
        ]
        mean = fmean(fracs) if fracs else 0.0
        lines.append(f"{spec}: mean frac {mean:.2f} over {len(fracs)} rung(s)")
    return lines


def render_report(rungs: Sequence[RungAggregate]) -> str:
    """The stdout table plus one summary line per spec."""
    rows = [_format_row(aggregate) for aggregate in rungs]
    widths = _widths(rows)
    lines = [_join(_HEADERS, widths), _join(["-" * width for width in widths], widths)]
    lines.extend(_join(row, widths) for row in rows)
    lines.extend(_spec_summaries(rungs))
    return "\n".join(lines)


def _stderr_progress(record: RequestRecord) -> None:
    print(
        f"{record.spec} rung={record.rung} {record.role} {record.attempt} "
        f"{record.status} {record.cached}/{record.prompt_total} {record.latency_ms:.0f}ms",
        file=sys.stderr,
        flush=True,
    )


# --- command line -------------------------------------------------------------


def default_targets_path() -> Path:
    """The packaged `targets.toml`; `--targets PATH` overrides it."""
    return Path(provibench.__file__).resolve().parent / "data" / "targets.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_poc",
        description="Probe prompt-cache behaviour on a few turns of a recorded trace.",
    )
    parser.add_argument("trace", help="trace file (one JSON request per line)")
    parser.add_argument("spec", nargs="+", help="run spec <target>:<model>[@provider,...]")
    parser.add_argument("--rungs", default="1", help="1-based turn indices, comma-separated")
    parser.add_argument("--repeats", default="1", help="warm repeats per rung, comma-separated")
    parser.add_argument("--gap", type=float, default=1.0, help="seconds between warm requests")
    parser.add_argument("--timeout", type=float, default=300.0, help="request timeout, seconds")
    parser.add_argument("--targets", default=None, help="targets.toml path")
    parser.add_argument("--output-file", default=None, help="write the JSON report to this file")
    return parser


async def _run(
    config: ProbeConfig,
    entries: Sequence[TraceEntry],
    api_keys: Mapping[str, str],
    state: ProbeState,
) -> ProbeRun:
    timeout = httpx.Timeout(config.timeout_s, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await run_probe(
            config, entries, api_keys, state=state, client=client, progress=_stderr_progress
        )


def _write_report(path: Path, run: ProbeRun) -> None:
    path.write_text(json.dumps(run.model_dump(), indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trace_path = Path(args.trace)
    try:
        targets = load_targets(Path(args.targets) if args.targets else default_targets_path())
        specs = [parse_run_spec(raw, targets) for raw in args.spec]
        api_keys = {spec.label: resolve_api_key(spec.target, os.environ) for spec in specs}
        rungs = parse_int_list(args.rungs)
        repeats = parse_int_list(args.repeats)
        entries = load_trace(trace_path)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = ProbeConfig(
        trace=trace_path,
        specs=specs,
        rungs=rungs,
        repeats=repeats,
        gap_s=args.gap,
        timeout_s=args.timeout,
    )
    state = ProbeState.start(trace_path)
    try:
        asyncio.run(_run(config, entries, api_keys, state))
    except ValueError as exc:  # an impossible rung, before any request was sent
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        # An interrupt or unexpected failure still writes the records already paid for.
        if args.output_file is not None and state.outcomes:
            _write_report(Path(args.output_file), state.build(config))

    run = state.build(config)
    print(render_report(run.rungs))
    return exit_code(run.requests)


if __name__ == "__main__":
    raise SystemExit(main())
