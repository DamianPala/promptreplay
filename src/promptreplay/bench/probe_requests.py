"""The probe's request path: one attempt, the retries, and the record it produces.

Every body a probe sends — the cold write, the warm reads, the streamed throughput request —
goes through here, so a retried request is byte-identical to the one it replaces and only
the last attempt's wall clock is recorded. A 429 or 503 is retried on the backoff schedule;
a 400 that means the recorded `thinking` parameter met a one-token `max_tokens` is retried
once with that key dropped, exactly as `replay` does it.

The streamed request parses its own response, so it hands the record builder a
`ParsedMessage` instead of letting it parse an `httpx.Response` a second time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter
from typing import Any

import httpx

from promptreplay.bench.probe_context import ProbeContext
from promptreplay.bench.probe_models import ProbeCall, ProbeOutcome, ProbeResult
from promptreplay.bench.replay import ReplayResult
from promptreplay.bench.requests import build_headers, parse_body, should_retry_without_thinking
from promptreplay.bench.sse import ParsedMessage
from promptreplay.bench.trace import TraceEntry

_RETRY_STATUSES = (429, 503)
_RETRY_DELAYS = (2.0, 5.0)
THINKING_DROPPED = "thinking param dropped"

type PostFn = Callable[..., Any]
"""How one attempt is sent. Both `bench.requests.post` and `bench.probe_stream.post_stream`
are awaited with the same four keywords and return a response-like object."""

type StreamFields = Callable[[Any, ParsedMessage], dict[str, Any]]
"""Extra record fields a streamed response contributes: TTFT, tok/s, the fingerprint."""

type RecordFn = Callable[..., ProbeOutcome]
"""How an attempt's result becomes a record. The streamed request supplies its own, because
its result carries timing the non-streaming path never sees."""

__all__ = ["THINKING_DROPPED", "default_record", "error_record", "outcome_from", "send"]


async def send(
    probe: ProbeContext,
    post_fn: PostFn,
    entry: TraceEntry,
    body: dict[str, Any],
    call: ProbeCall,
    *,
    drop_thinking: bool = False,
    record: RecordFn | None = None,
) -> ProbeOutcome:
    """Send one prepared body, retrying transient overloads, and report the outcome.

    `record` builds the outcome from the attempt's result; the default reads the response
    body here, while the streamed request has already read its own.
    """
    headers = build_headers(entry, probe.api_key)
    retries = 0
    attempt_body = body
    dropped = drop_thinking
    while True:
        started = perf_counter()
        try:
            resp = await post_fn(
                probe.client, probe.spec, headers, attempt_body, timeout_s=probe.opts.timeout_s
            )
        except httpx.HTTPError as exc:
            outcome = ProbeOutcome(
                error_record(
                    probe,
                    entry,
                    call,
                    started,
                    error=str(exc),
                    retries=retries,
                    thinking_dropped=dropped,
                ),
                thinking_dropped=dropped,
            )
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
        builder = record if record is not None else default_record
        outcome = builder(
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


def error_record(
    probe: ProbeContext,
    entry: TraceEntry,
    call: ProbeCall,
    started: float,
    *,
    error: str,
    retries: int,
    thinking_dropped: bool = False,
) -> ProbeResult:
    """The record of a request that never produced a response."""
    return ProbeResult(
        spec_label=probe.spec.label,
        rung=call.rung,
        role=call.role,
        attempt=call.attempt,
        seq=entry.seq,
        status=0,
        latency_ms=(perf_counter() - started) * 1000,
        retries=retries,
        note=THINKING_DROPPED if thinking_dropped else None,
        error=error,
    )


def default_record(
    probe: ProbeContext,
    entry: TraceEntry,
    resp: httpx.Response,
    call: ProbeCall,
    *,
    latency_ms: float,
    retries: int,
    thinking_dropped: bool = False,
) -> ProbeOutcome:
    """The record of an ordinary (non-streaming) response."""
    return outcome_from(
        probe,
        entry,
        resp,
        call,
        latency_ms=latency_ms,
        retries=retries,
        thinking_dropped=thinking_dropped,
    )


def outcome_from(  # noqa: PLR0913 (every keyword is an optional detail of the same record)
    probe: ProbeContext,
    entry: TraceEntry,
    resp: Any,
    call: ProbeCall,
    *,
    latency_ms: float,
    retries: int,
    thinking_dropped: bool = False,
    parsed: ParsedMessage | None = None,
    stream_fields: StreamFields | None = None,
) -> ProbeOutcome:
    """The record, and the replay result that will carry its generation lookup.

    A caller that already parsed the response passes `parsed`; otherwise the body is parsed
    here. `stream_fields` are merged onto the record last, so they win over the defaults.
    """
    parsed = parsed if parsed is not None else parse_body(resp)
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
        note=THINKING_DROPPED if thinking_dropped else None,
        error=parsed.error,
        **(stream_fields(resp, parsed) if stream_fields is not None else {}),
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
    return ProbeOutcome(record, replay, thinking_dropped=thinking_dropped)
