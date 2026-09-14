"""Probe protocol: measure a provider's prompt cache from a few real turns of a trace.

A rung is a 1-based turn `k` of the conversation. The cold request sends turn `k`, which
writes the cache; the warm requests send turn `k+1`, whose prompt starts with the same
bytes, so what they read back measures whether the provider cached the prefix and whether
the next request landed on a replica that has it. Each rung carries its own nonce, so rung
30 does not read what rung 1 wrote; a cold request that fails skips its rung, because
there is nothing cached to warm — a skipped rung is not a miss.

That is the whole measurement, and it costs a fraction of a full replay: the same ranking
from a handful of independent samples instead of 30 correlated ones. See
`docs/research/probe-poc.md` for the numbers behind the protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

from provibench.bench.enrich import append_note, enrich_openrouter
from provibench.bench.nonce import inject_nonce, new_run_hex, probe_nonce, require_stampable
from provibench.bench.probe_models import (
    ProbeOptions,
    ProbeResult,
    ProbeRun,
    Role,
    is_failed,
    is_served,
)
from provibench.bench.replay import ReplayOptions, ReplayResult, prepare_body
from provibench.bench.requests import build_headers, parse_body, post, should_retry_without_thinking
from provibench.bench.targets import RunSpec, resolve_api_key
from provibench.bench.trace import TraceEntry

_RETRY_STATUSES = (429, 503)
_RETRY_DELAYS = (2.0, 5.0)
_THINKING_DROPPED = "thinking param dropped"

__all__ = [
    "ProbeOptions",
    "ProbeResult",
    "ProbeRun",
    "is_failed",
    "is_served",
    "run_probe",
]
"""`bench.probe` is the protocol's home: the vocabulary is re-exported from `probe_models`."""


# --- the run ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Run:
    """Run-level context every request needs, resolved once."""

    run_hex: str
    warm: bool
    gap_s: float
    rungs: list[int]
    counts: list[int]

    def nonce(self, rung: int) -> str | None:
        """The rung's nonce, or `None` in warm mode, where no nonce is stamped."""
        return None if self.warm else probe_nonce(self.run_hex, rung)


@dataclass(frozen=True, slots=True)
class _Call:
    """Which request this is, for the record it produces."""

    rung: int
    role: Role
    attempt: int
    nonce: str | None


@dataclass(slots=True)
class _Probe:
    """The per-spec context shared by its requests."""

    spec: RunSpec
    opts: ReplayOptions
    api_key: str
    client: httpx.AsyncClient
    progress: Callable[[ProbeResult], None] | None = None


@dataclass(slots=True)
class _Outcome:
    """A record and the replay result that carries its generation lookup."""

    record: ProbeResult
    replay: ReplayResult | None = None
    thinking_dropped: bool = False
    """The request carried a `thinking` parameter that had to go for the replay to work."""


async def run_probe(
    specs: Sequence[RunSpec],
    entries: Sequence[TraceEntry],
    options: ProbeOptions,
    env: Mapping[str, str],
    *,
    on_progress: Callable[[ProbeResult], None] | None = None,
) -> ProbeRun:
    """Send every rung of every spec, one spec at a time, rungs in order."""
    # Resolve keys and check the nonce up front: both fail before any request is sent.
    api_keys = {spec.label: resolve_api_key(spec.target, env) for spec in specs}
    resolved = options.resolved(entries)
    rungs = resolved.rungs or []
    if not resolved.warm:
        require_stampable(
            (f"rung {rung} turn {turn}", entries[turn - 1].body)
            for rung in rungs
            for turn in (rung, rung + 1)
        )
    run = _Run(
        run_hex=new_run_hex(),
        warm=resolved.warm,
        gap_s=resolved.gap_s,
        rungs=resolved.rungs or [],
        counts=resolved.repeats,
    )
    records: dict[str, list[ProbeResult]] = {}
    timeout = httpx.Timeout(resolved.timeout_s, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for spec in specs:
            probe = _Probe(
                spec=spec,
                opts=resolved.replay_options(),
                api_key=api_keys[spec.label],
                client=client,
                progress=on_progress,
            )
            outcomes: list[_Outcome] = []
            await _probe_spec(probe, run, entries, outcomes)
            await _enrich(probe, outcomes)
            records[spec.label] = [outcome.record for outcome in outcomes]
    return ProbeRun(run_hex=run.run_hex, options=resolved, records=records)


async def _probe_spec(
    probe: _Probe, run: _Run, conversation: Sequence[TraceEntry], outcomes: list[_Outcome]
) -> None:
    """Send every rung of one spec; a rung whose cold request fails is skipped."""
    for rung, count in zip(run.rungs, run.counts, strict=True):
        nonce = run.nonce(rung)
        cold_entry = conversation[rung - 1]
        cold_call = _Call(rung, "cold", 0, nonce)
        cold = await _send(probe, cold_entry, _body(cold_entry, probe, nonce), cold_call)
        outcomes.append(cold)
        if not is_served(cold.record):
            continue  # nothing was cached, so there is nothing to warm
        warm_entry = conversation[rung]
        # A `thinking` parameter the cold request had to drop is dropped from the warm
        # bodies too, or the prefix they measure is not the one the cold request wrote.
        drop = cold.thinking_dropped
        warm_body = _body(warm_entry, probe, nonce, drop_thinking=drop)
        for attempt in range(1, count + 1):
            # A cache write can land after the response returns, so wait before every warm
            # attempt, the first one included.
            if run.gap_s > 0:
                await asyncio.sleep(run.gap_s)
            call = _Call(rung, "warm", attempt, nonce)
            outcome = await _send(probe, warm_entry, warm_body, call, drop_thinking=drop)
            outcomes.append(outcome)
            if outcome.thinking_dropped and not drop:
                # This repeat and every later one must send the same bytes as each other;
                # the repeats already sent keep their recorded outcome.
                drop = True
                warm_body = _body(warm_entry, probe, nonce, drop_thinking=True)


def _body(
    entry: TraceEntry, probe: _Probe, nonce: str | None, *, drop_thinking: bool = False
) -> dict[str, Any]:
    """The recorded body prepared for the probe, stamped with the rung's nonce."""
    prepared = prepare_body(entry.body, probe.spec, probe.opts)
    if drop_thinking:
        prepared.pop("thinking", None)
    return prepared if nonce is None else inject_nonce(prepared, nonce)


async def _send(
    probe: _Probe,
    entry: TraceEntry,
    body: dict[str, Any],
    call: _Call,
    *,
    drop_thinking: bool = False,
) -> _Outcome:
    """POST one prepared body, retrying transient overloads, and report the outcome.

    A 429 or 503 is retried on the backoff schedule; the bytes are identical across
    attempts, so resending is safe. A 400 that means the recorded `thinking` parameter met
    the one-token `max_tokens` is retried once with that key dropped, as replay does, and
    the caller carries the decision on to the rest of the rung. Only the last attempt's
    wall clock is recorded, so neither wait inflates the latency.
    """
    headers = build_headers(entry, probe.api_key)
    retries = 0
    attempt_body = body
    dropped = drop_thinking
    while True:
        started = perf_counter()
        try:
            resp = await post(
                probe.client, probe.spec, headers, attempt_body, timeout_s=probe.opts.timeout_s
            )
        except httpx.HTTPError as exc:
            failed = _error_record(
                probe,
                entry,
                call,
                started,
                error=str(exc),
                retries=retries,
                thinking_dropped=dropped,
            )
            outcome = _Outcome(failed, thinking_dropped=dropped)
            break
        if resp.status_code in _RETRY_STATUSES and retries < len(_RETRY_DELAYS):
            await asyncio.sleep(_RETRY_DELAYS[retries])
            retries += 1
            continue
        if (
            resp.status_code == 400
            and "thinking" in attempt_body
            and should_retry_without_thinking(resp.text)
        ):
            dropped = True
            attempt_body = {key: value for key, value in attempt_body.items() if key != "thinking"}
            continue
        outcome = _outcome_from(
            probe,
            entry,
            resp,
            call,
            latency_ms=(perf_counter() - started) * 1000,
            retries=retries,
            thinking_dropped=dropped,
        )
        break
    if probe.progress is not None:
        probe.progress(outcome.record)
    return outcome


def _error_record(
    probe: _Probe,
    entry: TraceEntry,
    call: _Call,
    started: float,
    *,
    error: str,
    retries: int,
    thinking_dropped: bool = False,
) -> ProbeResult:
    return ProbeResult(
        spec_label=probe.spec.label,
        rung=call.rung,
        role=call.role,
        attempt=call.attempt,
        seq=entry.seq,
        status=0,
        latency_ms=(perf_counter() - started) * 1000,
        retries=retries,
        note=_THINKING_DROPPED if thinking_dropped else None,
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
    thinking_dropped: bool = False,
) -> _Outcome:
    """The record, and the replay result that will carry its generation lookup."""
    parsed = parse_body(resp)
    usage = parsed.usage
    record = ProbeResult(
        spec_label=probe.spec.label,
        rung=call.rung,
        role=call.role,
        attempt=call.attempt,
        seq=entry.seq,
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
        retries=retries,
        note=_THINKING_DROPPED if thinking_dropped else None,
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
    return _Outcome(record, replay, thinking_dropped=thinking_dropped)


async def _enrich(probe: _Probe, outcomes: list[_Outcome]) -> None:
    """Run the OpenRouter generation lookup for a spec, best-effort."""
    if probe.spec.target.kind != "openrouter":
        return
    replays = [outcome.replay for outcome in outcomes if outcome.replay is not None]
    try:
        await enrich_openrouter(probe.spec, replays, probe.client, probe.api_key)
    except (httpx.HTTPError, ValueError) as exc:
        note = f"generation enrichment failed: {exc}"
        for outcome in outcomes:
            if is_served(outcome.record):
                outcome.record.note = note
        return
    for outcome in outcomes:
        _apply_enrichment(outcome)


def _native_cached(generation: dict[str, Any]) -> int | None:
    value = generation.get("native_tokens_cached")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _apply_enrichment(outcome: _Outcome) -> None:
    """Copy the generation lookup onto the record.

    The response's own usage stays in `cached`/`prompt_total`; the native count is kept
    beside it, so a divergence between what the gateway billed and what the endpoint
    reported is visible. Aggregation prefers the response and falls back to the native
    count, which is where a gateway that reports no cache at all still gets credited.
    """
    replay = outcome.replay
    if replay is None:
        return
    if replay.note:
        # Appended, not replaced: a dropped `thinking` parameter is already noted here.
        outcome.record.note = append_note(outcome.record.note, replay.note)
    if replay.generation is None:
        return
    outcome.record.provider = replay.provider
    outcome.record.generation = replay.generation
    outcome.record.native_tokens_cached = _native_cached(replay.generation)
