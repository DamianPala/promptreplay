"""Probe protocol: measure a provider's prompt cache from a few real turns of a trace.

A rung is a 1-based turn `k` of the conversation. The cold request sends turn `k`, which
writes the cache; the warm requests send turn `k+1`, whose prompt starts with the same
bytes, so what they read back measures whether the provider cached the prefix and whether
the next request landed on a replica that has it. Each rung carries its own nonce, so rung
30 does not read what rung 1 wrote; a cold request that fails skips its rung, because there
is nothing cached to warm — a skipped rung is not a miss.

Two requests ride along. The throughput request sends the warm body once more, streamed and
with a real output budget, which is where time-to-first-token, tokens per second and the
model fingerprint come from. The TTL re-reads (`--ttl`) resend the first rung's warm body
after a delay, which is what turns "it is cached" into "it is cached for at least this
long". See `docs/research/probe-poc.md` for the numbers behind the protocol.
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
from provibench.bench.probe_context import ProbeContext
from provibench.bench.probe_models import (
    ProbeCall,
    ProbeOptions,
    ProbeOutcome,
    ProbeResult,
    ProbeRun,
    is_failed,
    is_served,
)
from provibench.bench.probe_requests import send
from provibench.bench.probe_stream import build_stream_body, post_stream, stream_record
from provibench.bench.replay import ReplayResult, prepare_body
from provibench.bench.requests import post
from provibench.bench.targets import RunSpec, resolve_api_key
from provibench.bench.trace import TraceEntry

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


async def run_probe(
    specs: Sequence[RunSpec],
    entries: Sequence[TraceEntry],
    options: ProbeOptions,
    env: Mapping[str, str],
    *,
    on_progress: Callable[[ProbeResult], None] | None = None,
    parallel: int = 1,
) -> ProbeRun:
    """Send every rung of every spec, rungs in order, up to `parallel` specs at a time.

    A spec is sequential from its cold write to its last TTL read — the protocol is a
    timeline, and overlapping its own requests would measure nothing. Concurrency is only
    ever across specs, which are independent: they hit different providers, so neither the
    cache they write nor the wall clock they take can leak into another spec's numbers.
    What does move is latency: with `parallel` above one the prefill and TTFT medians sit
    next to requests from elsewhere on the same connection pool.
    """
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
    timeout = httpx.Timeout(resolved.timeout_s, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        gate = asyncio.Semaphore(max(1, parallel))
        probed = await asyncio.gather(
            *(
                _probe_one(
                    spec,
                    client=client,
                    gate=gate,
                    run=run,
                    entries=entries,
                    options=resolved,
                    api_key=api_keys[spec.label],
                    on_progress=on_progress,
                )
                for spec in specs
            )
        )
    return ProbeRun(run_hex=run.run_hex, options=resolved, records=dict(probed))


async def _probe_one(
    spec: RunSpec,
    *,
    client: httpx.AsyncClient,
    gate: asyncio.Semaphore,
    run: _Run,
    entries: Sequence[TraceEntry],
    options: ProbeOptions,
    api_key: str,
    on_progress: Callable[[ProbeResult], None] | None,
) -> tuple[str, list[ProbeResult]]:
    """One spec start to finish, holding one of the run's concurrency slots."""
    async with gate:
        probe = ProbeContext(
            spec=spec,
            opts=options.replay_options(),
            api_key=api_key,
            client=client,
            throughput=options.throughput,
            ttl_s=options.ttl_s,
            progress=on_progress,
        )
        outcomes: list[ProbeOutcome] = []
        await _probe_spec(probe, run, entries, outcomes)
        await _enrich(probe, outcomes)
    return spec.label, [outcome.record for outcome in outcomes]


@dataclass(frozen=True, slots=True)
class _Rung:
    """One rung's shared state: what the extra requests send, and when the cache was warm.

    `warm_done` is read the moment the last warm read returns, because that is the instant
    the TTL offsets are measured from; taking it later would add whatever ran in between.
    """

    rung: int
    count: int
    nonce: str | None
    entry: TraceEntry
    body: dict[str, Any]
    drop: bool
    warm_done: float


async def _probe_spec(
    probe: ProbeContext, run: _Run, conversation: Sequence[TraceEntry], outcomes: list[ProbeOutcome]
) -> None:
    """Send every rung of one spec; a rung whose cold request fails is skipped."""
    ttl_rung: _Rung | None = None
    for rung, count in zip(run.rungs, run.counts, strict=True):
        nonce = run.nonce(rung)
        cold_entry = conversation[rung - 1]
        cold = await _send(
            probe, cold_entry, _body(cold_entry, probe, nonce), ProbeCall(rung, "cold", 0, nonce)
        )
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
            call = ProbeCall(rung, "warm", attempt, nonce)
            outcome = await _send(probe, warm_entry, warm_body, call, drop_thinking=drop)
            outcomes.append(outcome)
            if outcome.thinking_dropped and not drop:
                # This repeat and every later one must send the same bytes as each other;
                # the repeats already sent keep their recorded outcome.
                drop = True
                warm_body = _body(warm_entry, probe, nonce, drop_thinking=True)
        state = _Rung(rung, count, nonce, warm_entry, warm_body, drop, perf_counter())
        if probe.throughput:
            outcomes.append(await _throughput(probe, state))
        if probe.ttl_s and ttl_rung is None:
            # TTL rides the first rung that was actually served: a rung whose cold request
            # failed wrote nothing to re-read. It is sequential per spec, like everything
            # else, so it waits here rather than after the rungs below.
            ttl_rung = state
            outcomes.extend(await _ttl_phase(probe, state))


async def _throughput(probe: ProbeContext, state: _Rung) -> ProbeOutcome:
    """One streamed generation on the warm body: same prefix, a real `max_tokens`."""

    async def post_body(
        client: httpx.AsyncClient,
        spec: RunSpec,
        headers: dict[str, str],
        attempt_body: dict[str, Any],
        *,
        timeout_s: float,
    ) -> Any:
        return await post_stream(client, spec, headers, attempt_body, timeout_s=timeout_s)

    def record(
        _probe: ProbeContext,
        _entry: TraceEntry,
        response: Any,
        call: ProbeCall,
        *,
        latency_ms: float,
        retries: int,
        thinking_dropped: bool = False,
    ) -> ProbeOutcome:
        return stream_record(
            probe,
            state.entry,
            response,
            call,
            latency_ms=latency_ms,
            retries=retries,
            thinking_dropped=thinking_dropped,
        )

    return await send(
        probe,
        post_body,
        state.entry,
        build_stream_body(state.body),
        ProbeCall(state.rung, "stream", 0, state.nonce),
        drop_thinking=state.drop,
        record=record,
    )


async def _ttl_phase(probe: ProbeContext, state: _Rung) -> list[ProbeOutcome]:
    """Re-send one rung's warm body at each `--ttl` offset past its last warm read.

    Each read waits until the wall clock says its offset has passed, counted from the moment
    that rung's last warm read returned. Anything sent in between — the throughput request,
    a retry — is therefore part of the elapsed time rather than added on top of it, so a
    `--ttl 60` read lands 60 s after the cache was warm, not 60 s plus a generation. Where
    it really landed is recorded next to the offset that was asked for: a machine that slept
    through the wait, or a busy event loop, shows up as a late read rather than a wrong one.
    """
    outcomes: list[ProbeOutcome] = []
    for offset in probe.ttl_s or []:
        remaining = state.warm_done + offset - perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        call = ProbeCall(state.rung, "ttl", offset, state.nonce)
        sent_at = perf_counter() - state.warm_done
        outcome = await _send(probe, state.entry, state.body, call)
        outcome.record.offset_actual_s = sent_at
        outcomes.append(outcome)
    return outcomes


async def _send(
    probe: ProbeContext,
    entry: TraceEntry,
    body: dict[str, Any],
    call: ProbeCall,
    *,
    drop_thinking: bool = False,
) -> ProbeOutcome:
    """One non-streaming request of the protocol."""
    return await send(probe, post, entry, body, call, drop_thinking=drop_thinking)


def _body(
    entry: TraceEntry, probe: ProbeContext, nonce: str | None, *, drop_thinking: bool = False
) -> dict[str, Any]:
    """The recorded body prepared for the probe, stamped with the rung's nonce."""
    prepared = prepare_body(entry.body, probe.spec, probe.opts)
    if drop_thinking:
        prepared.pop("thinking", None)
    return prepared if nonce is None else inject_nonce(prepared, nonce)


async def _enrich(probe: ProbeContext, outcomes: list[ProbeOutcome]) -> None:
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


def _apply_enrichment(outcome: ProbeOutcome) -> None:
    """Copy the generation lookup onto the record.

    The response's own usage stays in `cached`/`prompt_total`; the native count is kept
    beside it, so a divergence between what the gateway billed and what the endpoint
    reported is visible. Aggregation prefers the response and falls back to the native
    count, which is where a gateway that reports no cache at all still gets credited.
    """
    replay: ReplayResult | None = outcome.replay
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
