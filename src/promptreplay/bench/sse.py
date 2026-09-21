"""Extract message metadata and usage from Anthropic Messages API responses.

Two readers of the same wire format live here. `parse_event_stream` folds a whole SSE body
at once, which is what a non-streaming call sees. `read_stream` drains a streaming response
event by event, stamping each arrival with the caller's clock: that is what turns a streamed
generation into a time-to-first-token, a generation rate and a model fingerprint, the
numbers the probe's throughput request exists for.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field

from promptreplay.bench.trace import Usage

_SSE_SEPARATOR = "\n\n"
_DELTA_TYPES = {
    "thinking_delta": "thinking",
    "text_delta": "text",
    "input_json_delta": "partial_json",
}
"""Where each delta kind carries its text: a tool call streams its arguments as JSON."""
_TERMINAL_EVENTS = ("message_stop", "error")
"""The events after which a stream carries nothing more, whatever the socket does."""
_ERROR_STATUS = 400
"""From here up a response is a refusal; its body is read as the error it is."""
FINGERPRINT_TOKENS = 16
"""Output tokens taken in order to identify the model behind an endpoint."""


class ParsedMessage(BaseModel):
    message_id: str | None = None
    model: str | None = None
    provider: str | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


def _merge_usage(base: dict[str, Any], extra: dict[str, Any] | None) -> None:
    for key, value in (extra or {}).items():
        if isinstance(value, int):
            base[key] = value


def parse_json_message(data: dict[str, Any], *, status: int | None = None) -> ParsedMessage:
    """One JSON body as a message; with `status` an error status, the body is the error.

    A provider spells its refusals differently — Anthropic wraps one in `error`, OpenRouter's
    compatibility layer sends the error object itself (`{"type": "not_found_error", ...}`) —
    so the shape alone cannot tell a refusal from an answer. The status can, and a refusal's
    whole body is kept: it is where the reason a request was rejected is written.
    """
    if data.get("type") == "error" or "error" in data:
        err = data.get("error", data)
        return ParsedMessage(error=json.dumps(err)[:500])
    if status is not None and status >= _ERROR_STATUS:
        return ParsedMessage(error=json.dumps(data)[:500])
    return ParsedMessage(
        message_id=data.get("id"),
        model=data.get("model"),
        provider=data.get("provider"),
        stop_reason=data.get("stop_reason"),
        usage=Usage.from_api(data.get("usage")),
    )


def _as_dict(value: object) -> dict[str, Any]:
    """`value` as a string-keyed dict, or an empty one when it isn't."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def fold_event(out: ParsedMessage, usage: dict[str, Any], data: dict[str, Any]) -> None:
    """Apply one interpreted event to the message being accumulated."""
    kind = data.get("type")
    if kind == "message_start":
        msg = _as_dict(data.get("message"))
        out.message_id = msg.get("id")
        out.model = msg.get("model")
        out.provider = msg.get("provider") or data.get("provider")
        _merge_usage(usage, msg.get("usage"))
    elif kind == "message_delta":
        delta = _as_dict(data.get("delta"))
        out.stop_reason = delta.get("stop_reason") or out.stop_reason
        _merge_usage(usage, data.get("usage"))
    elif kind == "error":
        out.error = json.dumps(data.get("error", data))[:500]


def interpret_event(payload: str) -> dict[str, Any] | None:
    """One SSE `data:` payload as a message event, or `None` for a marker that isn't one."""
    if not payload or payload == "[DONE]":
        return None
    try:
        value: object = json.loads(payload)
    except json.JSONDecodeError:
        return None
    data = _as_dict(value)
    return data or None


def parse_event_stream(raw: bytes) -> ParsedMessage:
    """Fold an SSE stream into one ParsedMessage.

    message_start carries id/model and the prompt-side usage; message_delta carries
    output_tokens (and on some gateways a final input_tokens correction).
    """
    out = ParsedMessage()
    usage: dict[str, Any] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = interpret_event(line[5:].strip())
        if data is not None:
            fold_event(out, usage, data)
    out.usage = Usage.from_api(usage)
    return out


def parse_response(content_type: str, raw: bytes, *, status: int | None = None) -> ParsedMessage:
    if "text/event-stream" in content_type:
        return parse_event_stream(raw)
    try:
        return parse_json_message(json.loads(raw or b"{}"), status=status)
    except json.JSONDecodeError:
        return ParsedMessage(error=raw[:500].decode("utf-8", "replace"))


async def iter_stream_events(
    response: httpx.Response, *, idle_timeout_s: float | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Yield each event of an SSE response as it arrives, interpreting the `data:` lines.

    Reading stops at the event that ends a stream, so a connection the server leaves open
    cannot hold the run: the response is closed by whoever opened it, without waiting for
    the socket. `idle_timeout_s` bounds the other silence — a stream that sends nothing at
    all — and raises `httpx.ReadTimeout` so the caller's request path handles it as any
    other timeout.
    """
    chunks = response.aiter_text().__aiter__()
    buffer = ""
    while True:
        try:
            chunk = await _next_chunk(chunks, idle_timeout_s)
        except StopAsyncIteration:
            break
        buffer += chunk
        while _SSE_SEPARATOR in buffer:
            block, buffer = buffer.split(_SSE_SEPARATOR, 1)
            data = _event_data(block)
            if data is not None:
                yield data
                if _ends_stream(data):
                    return
    data = _event_data(buffer)
    if data is not None:
        yield data


async def _next_chunk(chunks: AsyncIterator[str], idle_timeout_s: float | None) -> str:
    """The next body chunk, or a `ReadTimeout` when the stream has gone quiet."""
    if idle_timeout_s is None:
        return await chunks.__anext__()
    try:
        return await asyncio.wait_for(chunks.__anext__(), idle_timeout_s)
    except TimeoutError as exc:
        raise httpx.ReadTimeout(f"stream stalled: no event for {idle_timeout_s:g}s") from exc


def _ends_stream(data: dict[str, Any]) -> bool:
    """Whether this event is the last one a stream will carry."""
    kind = data.get("type")
    return kind in _TERMINAL_EVENTS


def _event_data(block: str) -> dict[str, Any] | None:
    """The first `data:` line of one event block, as an interpreted event."""
    for line in block.splitlines():
        if line.startswith("data:"):
            return interpret_event(line[5:].strip())
    return None


class StreamStats:
    """What a streamed response looked like as it arrived, and what it generated.

    Every time is seconds since the caller's clock origin, so the caller passes a clock
    that measures from the request's own start (`bench.probe_stream` does): a time here is
    already the time-to-first-token. `done_at` is the event named `message_stop`, falling
    back to the last event seen, because a gateway that omits it still ends the stream.
    """

    def __init__(self, *, stop_after_tokens: int = FINGERPRINT_TOKENS) -> None:
        self.first_delta_at: float | None = None
        self.stop_at: float | None = None
        self.output_tokens: int | None = None
        self.pieces: list[str] = []
        self.deltas = 0
        """How many deltas carried content; one flush of a whole answer counts as one."""
        self.stop_after_tokens = stop_after_tokens

    def on_event(self, data: dict[str, Any], at: float) -> None:
        kind = data.get("type")
        if kind == "content_block_delta":
            if self.first_delta_at is None:
                self.first_delta_at = at
            self._add_delta(_as_dict(data.get("delta")))
        elif kind == "message_delta":
            output = _as_dict(data.get("usage")).get("output_tokens")
            if isinstance(output, int) and not isinstance(output, bool):
                self.output_tokens = output
        elif kind == "message_stop":
            self.stop_at = at

    def on_done(self, at: float) -> None:
        """The stream ended; without a `message_stop` this is the last event's time."""
        if self.stop_at is None:
            self.stop_at = at

    @property
    def ttft_s(self) -> float | None:
        """Seconds from the caller's clock origin — a request's start — to the first delta."""
        return self.first_delta_at

    @property
    def gen_tok_s(self) -> float | None:
        """Output tokens ÷ the time spent generating them; `None` when that is unknown."""
        if self.first_delta_at is None or self.stop_at is None or self.output_tokens is None:
            return None
        elapsed = self.stop_at - self.first_delta_at
        return self.output_tokens / elapsed if elapsed > 0 else None

    @property
    def text(self) -> str:
        """Every delta's text, joined with one space so no two pieces can fuse.

        A gateway is free to end a delta mid-word and start the next one continuing it; the
        space costs nothing where there already was one, and it is what keeps the last
        thinking token from running into the first text token.
        """
        return " ".join(self.pieces)

    @property
    def fingerprint(self) -> str | None:
        """The first `stop_after_tokens` output tokens, in arrival order, as one string.

        A stream carries text, not token boundaries, so the count is taken at whitespace:
        deterministic for every gateway, and the point is comparing one endpoint's string
        with another's, not reproducing a tokenizer.
        """
        tokens = self.text.split()[: self.stop_after_tokens]
        return " ".join(tokens) if tokens else None

    def _add_delta(self, delta: dict[str, Any]) -> None:
        """Collect a thinking or text delta; the two carry their text under different keys."""
        kind = delta.get("type")
        key = _DELTA_TYPES.get(kind) if isinstance(kind, str) else None
        if key is None:
            return
        text = delta.get(key)
        if isinstance(text, str) and text:
            self.pieces.append(text)
            self.deltas += 1


async def read_stream(
    response: httpx.Response,
    clock: Callable[[], float],
    *,
    stop_after_tokens: int = FINGERPRINT_TOKENS,
    idle_timeout_s: float | None = None,
) -> tuple[ParsedMessage, StreamStats]:
    """Drain an SSE response into one message, with the stream's own arrival statistics."""
    stats = StreamStats(stop_after_tokens=stop_after_tokens)
    out = ParsedMessage()
    usage: dict[str, Any] = {}
    async for data in iter_stream_events(response, idle_timeout_s=idle_timeout_s):
        fold_event(out, usage, data)
        stats.on_event(data, clock())
    stats.on_done(clock())
    out.usage = Usage.from_api(usage)
    return out, stats
