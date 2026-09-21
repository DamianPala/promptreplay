"""The probe's streamed throughput request: one generation per rung, timed as it arrives.

The warm reads answer "does this provider cache my prompt"; this answers the other two
questions: how fast the endpoint serves a token, and whether the model behind it is the one
that was asked for. It sends the same body the warm reads sent, once, with a real
`max_tokens` and temperature 0, and reads the response event by event so the first content
delta and the last one are both stamped. Two providers serving the same weights return the
same first tokens; a quantized or substituted endpoint returns different ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

from promptreplay.bench.enrich import append_note
from promptreplay.bench.probe_context import ProbeContext
from promptreplay.bench.probe_models import ProbeCall, ProbeOutcome
from promptreplay.bench.probe_requests import outcome_from
from promptreplay.bench.sse import ParsedMessage, StreamStats, parse_response, read_stream
from promptreplay.bench.trace import TraceEntry

STREAM_MAX_TOKENS = 256
"""The throughput request's output ceiling: enough to time generation, cheap to buy."""

_ERROR_STATUS = 400
"""From here up a streaming response is an error payload, not a stream to read."""

BURST_NOTE = "burst"
"""The record note of a stream that delivered its answer in one flush; see `StreamResult`."""

_WINDOW_MS = 250.0
"""Shortest generation window a rate can be read from: below it the divisor is noise."""

_DELTAS_AFTER_FIRST = 3
"""Content deltas a stream must send past its first for the window to divide anything."""

__all__ = [
    "BURST_NOTE",
    "STREAM_MAX_TOKENS",
    "StreamResult",
    "build_stream_body",
    "post_stream",
    "stream_record",
]


@dataclass(slots=True)
class StreamResult:
    """A streamed response: its message, its arrival statistics, and its wall clock."""

    message: ParsedMessage
    stats: StreamStats
    status: int
    latency_ms: float
    started: float

    @property
    def ttft_ms(self) -> float | None:
        """The wait for the first content delta, in milliseconds."""
        first = self.stats.first_delta_at
        return None if first is None else first * 1000

    @property
    def burst(self) -> bool:
        """Whether the answer arrived in one flush, which no rate can be read from.

        A provider that buffers a tool-call JSON and flushes it after the first token shows
        a generation window of a few milliseconds with one or two deltas in it; dividing a
        hundred tokens by that gives five figures of tokens per second and measures the
        buffer, not the model. The window has to be long enough to time a token in, and the
        content has to have arrived in more than one piece — the first delta plus three.

        A stream that produced no rate at all — a refused request, a stream that died before
        its first delta — is not a burst: there is no number to distrust, and calling it one
        would put "delivered in one flush" on a generation that never happened.
        """
        if self.stats.gen_tok_s is None:
            return False
        window_ms = self.latency_ms - (self.ttft_ms or 0.0)
        return window_ms < _WINDOW_MS or self.stats.deltas < _DELTAS_AFTER_FIRST + 1

    @property
    def gen_tok_s(self) -> float | None:
        """Output tokens per second over the generation window; `None` for a burst."""
        return None if self.burst else self.stats.gen_tok_s

    @property
    def status_code(self) -> int:
        """So a streamed attempt reads like any other response in the retry path."""
        return self.status

    @property
    def text(self) -> str:
        return self.message.error or ""


def build_stream_body(warm_body: Mapping[str, Any]) -> dict[str, Any]:
    """The warm body with a real output budget and no sampling: same bytes, more tokens.

    Built from the warm body rather than from the recording, so the prefix the streamed
    request reads is byte-identical to the one the warm reads read — sending anything else
    would make its own cache hit unmeasurable.
    """
    body = dict(warm_body)
    body["stream"] = True
    body["max_tokens"] = STREAM_MAX_TOKENS
    body["temperature"] = 0
    return body


async def post_stream(
    client: httpx.AsyncClient,
    spec: Any,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout_s: float,
) -> StreamResult:
    """POST one streaming body and drain it, timing every event with `perf_counter`.

    The response is opened with `stream=True` and read inside the same context manager, so
    a connection that fails or dies midway raises here, where the caller's retry lives. An
    error status is not a stream at all, so its body is read whole and parsed as the JSON
    error it is: that is what gives the record its error text and lets the caller's
    `thinking` retry see the 400 it is looking for. Reading stops at `message_stop`, and a
    stream that goes quiet for `timeout_s` raises rather than waiting on the socket, so one
    stuck endpoint cannot hold the whole run.
    """
    started = perf_counter()

    def since_start() -> float:
        """The clock `read_stream` stamps events with: seconds since this request began."""
        return perf_counter() - started

    async with client.stream(
        "POST", spec.target.url, headers=headers, json=body, timeout=timeout_s
    ) as response:
        status = response.status_code
        if status >= _ERROR_STATUS:
            await response.aread()
            message = parse_response(
                response.headers.get("content-type", ""), response.content, status=status
            )
            stats = StreamStats()
            stats.on_done(since_start())
        else:
            message, stats = await read_stream(response, since_start, idle_timeout_s=timeout_s)
    return StreamResult(
        message=message,
        stats=stats,
        status=status,
        latency_ms=(perf_counter() - started) * 1000,
        started=started,
    )


def stream_record(
    probe: ProbeContext,
    entry: TraceEntry,
    response: Any,
    call: ProbeCall,
    *,
    latency_ms: float,
    retries: int,
    thinking_dropped: bool = False,
) -> ProbeOutcome:
    """The throughput request's outcome: the stream's timing and fingerprint, plus its usage.

    `response` is the `StreamResult` this module posted; the record and the replay result are
    built by the shared path from the message it already parsed, so the OpenRouter generation
    lookup and the costing treat a streamed request like any other.
    """
    result: StreamResult = response
    outcome = outcome_from(
        probe,
        entry,
        result,
        call,
        latency_ms=latency_ms,
        retries=retries,
        thinking_dropped=thinking_dropped,
        parsed=result.message,
        stream_fields=lambda _resp, _parsed: {
            "ttft_ms": result.ttft_ms,
            "gen_tok_s": result.gen_tok_s,
            "fingerprint": result.stats.fingerprint,
        },
    )
    if result.burst:
        # Noted on the record because it is a property of this one response: the summary
        # names the rungs it happened on, and the raw jsonl says it here.
        outcome.record.note = append_note(outcome.record.note, BURST_NOTE)
    return outcome
