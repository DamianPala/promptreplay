"""Recording reverse proxy: sits between a harness and its API, writes a trace."""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from provibench.bench.sse import parse_response
from provibench.bench.trace import RecordedResponse, TraceEntry, append_entry, conversation_key

# Hop-by-hop plus everything that would lie once we re-frame the body. Dropping
# accept-encoding makes upstream send identity, so the bytes we tee are the bytes
# the harness receives and the SSE parser needs no decompression.
_DROP_REQ = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
_DROP_RESP = {"content-length", "connection", "transfer-encoding", "content-encoding"}
_KEEP_HEADERS = ("anthropic-version", "anthropic-beta")


def _default_log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


@dataclass(slots=True)
class _RequestContext:
    """Request-side facts captured before the response starts streaming back."""

    path: str
    query: str
    headers: dict[str, str]


@dataclass(slots=True)
class _CapturedResponse:
    """The raw response bytes and timing collected while teeing the upstream stream."""

    raw: bytes
    latency_ms: float
    ttft_ms: float | None


class Recorder:
    def __init__(
        self, upstream: str, trace_path: Path, *, log: Callable[[str], None] | None = None
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.trace_path = trace_path
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        self.seq = itertools.count(1)
        self.lock = asyncio.Lock()
        self.log = log or _default_log

    async def handle(self, request: Request) -> Response:
        body = await request.body()
        path = "/" + request.path_params.get("path", "")
        query = request.url.query
        url = f"{self.upstream}{path}" + (f"?{query}" if query else "")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQ}
        upstream_req = self.client.build_request(request.method, url, headers=headers, content=body)
        started = perf_counter()
        upstream_resp = await self.client.send(upstream_req, stream=True)

        parsed_body = _parse_messages_body(body) if path.endswith("/v1/messages") else None
        resp_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in _DROP_RESP
        }
        request_ctx = _RequestContext(path=path, query=query, headers=headers)
        return StreamingResponse(
            self._tee(upstream_resp, started, parsed_body, request_ctx),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )

    async def _tee(
        self,
        upstream_resp: httpx.Response,
        started: float,
        parsed_body: dict[str, Any] | None,
        request: _RequestContext,
    ) -> AsyncIterator[bytes]:
        chunks: list[bytes] = []
        ttft: float | None = None
        try:
            async for chunk in upstream_resp.aiter_raw():
                if ttft is None:
                    ttft = (perf_counter() - started) * 1000
                chunks.append(chunk)
                yield chunk
        finally:
            await upstream_resp.aclose()
            if parsed_body is not None:
                latency = (perf_counter() - started) * 1000
                captured = _CapturedResponse(raw=b"".join(chunks), latency_ms=latency, ttft_ms=ttft)
                await self._record(parsed_body, request, upstream_resp, captured)

    async def _record(
        self,
        body: dict[str, Any],
        request: _RequestContext,
        upstream_resp: httpx.Response,
        captured: _CapturedResponse,
    ) -> None:
        parsed = parse_response(upstream_resp.headers.get("content-type", ""), captured.raw)
        entry = TraceEntry(
            seq=next(self.seq),
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            path=request.path,
            query=request.query,
            headers={k: v for k, v in request.headers.items() if k.lower() in _KEEP_HEADERS},
            body=body,
            conversation=conversation_key(body),
            response=RecordedResponse(
                status=upstream_resp.status_code,
                latency_ms=round(captured.latency_ms, 1),
                ttft_ms=round(captured.ttft_ms, 1) if captured.ttft_ms is not None else None,
                message_id=parsed.message_id,
                model=parsed.model,
                provider=parsed.provider,
                stop_reason=parsed.stop_reason,
                usage=parsed.usage,
                error=parsed.error,
            ),
        )
        async with self.lock:
            append_entry(self.trace_path, entry)
        self.log(_progress_line(entry, captured.latency_ms))


def _progress_line(entry: TraceEntry, latency_ms: float) -> str:
    response = entry.response
    usage = response.usage if response else None
    return (
        f"#{entry.seq:<3} conv={entry.conversation} status={response.status if response else '?'} "
        f"prompt={usage.prompt_total if usage else '?'} "
        f"cached={usage.cache_read_input_tokens if usage else '?'} "
        f"write={usage.cache_creation_input_tokens if usage else '?'} "
        f"out={usage.output_tokens if usage else '?'} "
        f"ttft={response.ttft_ms if response else '?'}ms total={latency_ms:.0f}ms"
    )


def _parse_messages_body(body: bytes) -> dict[str, Any] | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "messages" in data:
        return cast("dict[str, Any]", data)
    return None


def serve(
    upstream: str,
    trace_path: Path,
    host: str,
    port: int,
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    """Run the recording proxy until interrupted; uvicorn installs its own SIGINT handler."""
    recorder = Recorder(upstream, trace_path, log=log)
    app = Starlette(
        routes=[
            Route(
                "/{path:path}",
                recorder.handle,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            )
        ]
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
