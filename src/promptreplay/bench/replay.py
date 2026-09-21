"""Replay engine: send recorded requests to a target, measure cache/cost, persist runs."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field

from promptreplay.bench.enrich import enrich_anthropic, enrich_openrouter
from promptreplay.bench.nonce import inject_nonce, require_stampable, run_nonce
from promptreplay.bench.prices import PriceTable
from promptreplay.bench.pricing import CostBreakdown
from promptreplay.bench.requests import (
    build_headers,
    parse_body,
    post,
    should_retry_without_thinking,
)
from promptreplay.bench.targets import RunSpec, resolve_api_key
from promptreplay.bench.trace import TraceEntry, Usage

_STRIPPED_BLOCK_TYPES = {"thinking", "redacted_thinking"}


class ReplayOptions(BaseModel):
    max_tokens: int = 1
    delay_s: float = 0.0
    strip_thinking: bool = False
    timeout_s: float = 300.0
    warm: bool = False
    """Skip the run nonce: measure the cache as found instead of writing a fresh one."""


class ReplayResult(BaseModel):
    seq: int
    turn: int
    status: int
    latency_ms: float
    message_id: str | None = None
    model: str | None = None
    provider: str | None = None
    requested_providers: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    prompt_total: int
    cached: int
    cache_write: int
    output_tokens: int
    cost: CostBreakdown | None = None
    billed_total: float | None = None
    cache_discount: float | None = None
    error: str | None = None
    note: str | None = None
    generation: dict[str, Any] | None = None


def _is_stripped_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    return cast("dict[str, Any]", block).get("type") in _STRIPPED_BLOCK_TYPES


def _strip_thinking(body: dict[str, Any]) -> None:
    messages: list[dict[str, Any]] = body.get("messages") or []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        blocks = cast("list[Any]", content)
        kept: list[Any] = [block for block in blocks if not _is_stripped_block(block)]
        msg["content"] = kept if kept else [{"type": "text", "text": " "}]


def prepare_body(body: dict[str, Any], spec: RunSpec, opts: ReplayOptions) -> dict[str, Any]:
    """Deep-copy the recorded body and apply only what replay needs to change."""
    prepared = copy.deepcopy(body)
    prepared["model"] = spec.model
    prepared["stream"] = False
    prepared["max_tokens"] = opts.max_tokens
    if spec.providers:
        prepared["provider"] = {"only": list(spec.providers), "allow_fallbacks": False}
    if opts.strip_thinking:
        _strip_thinking(prepared)
    return prepared


# The probe PoC script (`scripts/probe_poc.py`) reaches into this module for the request
# plumbing it shares with `replay`; the names stay for that record.
_build_headers = build_headers
_parse_body = parse_body


def _empty_result(
    entry: TraceEntry, turn: int, spec: RunSpec, latency_ms: float, *, error: str, note: str | None
) -> ReplayResult:
    return ReplayResult(
        seq=entry.seq,
        turn=turn,
        status=0,
        latency_ms=latency_ms,
        requested_providers=spec.providers,
        usage=Usage(),
        prompt_total=0,
        cached=0,
        cache_write=0,
        output_tokens=0,
        error=error,
        note=note,
    )


async def _post(
    client: httpx.AsyncClient,
    spec: RunSpec,
    headers: dict[str, str],
    body: dict[str, Any],
    opts: ReplayOptions,
) -> httpx.Response:
    return await post(client, spec, headers, body, timeout_s=opts.timeout_s)


async def _replay_turn(
    entry: TraceEntry,
    turn: int,
    spec: RunSpec,
    opts: ReplayOptions,
    api_key: str,
    *,
    client: httpx.AsyncClient,
    nonce: str | None = None,
) -> ReplayResult:
    prepared = prepare_body(entry.body, spec, opts)
    if nonce is not None:
        prepared = inject_nonce(prepared, nonce)
    headers = build_headers(entry, api_key)
    note: str | None = None
    started = perf_counter()

    try:
        resp = await _post(client, spec, headers, prepared, opts)
    except httpx.HTTPError as exc:
        elapsed = (perf_counter() - started) * 1000
        return _empty_result(entry, turn, spec, elapsed, error=str(exc), note=None)

    if resp.status_code == 400 and should_retry_without_thinking(resp.text):
        prepared.pop("thinking", None)
        note = "thinking param dropped"
        try:
            resp = await _post(client, spec, headers, prepared, opts)
        except httpx.HTTPError as exc:
            elapsed = (perf_counter() - started) * 1000
            return _empty_result(entry, turn, spec, elapsed, error=str(exc), note=note)

    latency_ms = (perf_counter() - started) * 1000
    parsed = parse_body(resp)
    usage = parsed.usage
    return ReplayResult(
        seq=entry.seq,
        turn=turn,
        status=resp.status_code,
        latency_ms=latency_ms,
        message_id=parsed.message_id,
        model=parsed.model,
        provider=parsed.provider,
        requested_providers=spec.providers,
        usage=usage,
        prompt_total=usage.prompt_total,
        cached=usage.cache_read_input_tokens,
        cache_write=usage.cache_creation_input_tokens,
        output_tokens=usage.output_tokens,
        error=parsed.error,
        note=note,
    )


# The probe PoC script (`scripts/probe_poc.py`) imports the enrichment under its old name;
# the engine itself calls `enrich_openrouter`.
_enrich_openrouter = enrich_openrouter


async def replay_run(
    spec: RunSpec,
    entries: list[TraceEntry],
    opts: ReplayOptions,
    api_key: str,
    *,
    client: httpx.AsyncClient,
    run_hex: str,
    on_progress: Callable[[ReplayResult], None] | None = None,
    table: PriceTable | None = None,
) -> list[ReplayResult]:
    nonce = None if opts.warm else run_nonce(run_hex)
    results: list[ReplayResult] = []
    for turn, entry in enumerate(entries, start=1):
        result = await _replay_turn(entry, turn, spec, opts, api_key, client=client, nonce=nonce)
        results.append(result)
        if on_progress is not None:
            on_progress(result)
        if opts.delay_s and turn < len(entries):
            await asyncio.sleep(opts.delay_s)

    if spec.target.kind == "openrouter":
        await enrich_openrouter(spec, results, client, api_key)
    else:
        enrich_anthropic(spec, results, table)
    return results


async def replay_all(
    specs: list[RunSpec],
    entries: list[TraceEntry],
    opts: ReplayOptions,
    env: Mapping[str, str],
    *,
    run_hex: str,
    on_progress: Callable[[ReplayResult], None] | None = None,
    table: PriceTable | None = None,
) -> dict[str, list[ReplayResult]]:
    # Resolve keys and check the nonce up front: both fail before any request is sent.
    api_keys = {spec.label: resolve_api_key(spec.target, env) for spec in specs}
    if not opts.warm:
        require_stampable(
            (f"turn {turn}", entry.body) for turn, entry in enumerate(entries, start=1)
        )
    timeout = httpx.Timeout(opts.timeout_s, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        outcomes = await asyncio.gather(
            *(
                replay_run(
                    spec,
                    entries,
                    opts,
                    api_keys[spec.label],
                    client=client,
                    run_hex=run_hex,
                    on_progress=on_progress,
                    table=table,
                )
                for spec in specs
            )
        )
    return {spec.label: result for spec, result in zip(specs, outcomes, strict=True)}


class RunRef(BaseModel):
    label: str
    slug: str
    target: str
    model: str
    providers: list[str] = Field(default_factory=list)
    file: str
    kind: str = "anthropic"
    """The target's kind at run time, so a re-read report knows which spec is the reference."""


class RunMeta(BaseModel):
    protocol: str = "full"
    trace: str
    conversation: str
    created: str
    run_hex: str | None = None
    options: ReplayOptions
    runs: list[RunRef]


def write_run(
    runs_dir: Path,
    trace_name: str,
    conversation: str,
    specs: list[RunSpec],
    opts: ReplayOptions,
    *,
    results: dict[str, list[ReplayResult]],
    run_hex: str | None = None,
) -> Path:
    created = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / trace_name / created
    run_dir.mkdir(parents=True, exist_ok=True)

    refs: list[RunRef] = []
    for spec in specs:
        file_name = f"{spec.slug}.jsonl"
        with (run_dir / file_name).open("w", encoding="utf-8") as fh:
            for result in results[spec.label]:
                fh.write(result.model_dump_json() + "\n")
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

    meta = RunMeta(
        trace=trace_name,
        conversation=conversation,
        created=created,
        run_hex=run_hex,
        options=opts,
        runs=refs,
    )
    (run_dir / "run.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run_dir


def load_run(run_dir: Path) -> tuple[RunMeta, dict[str, list[ReplayResult]]]:
    meta = RunMeta.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    results: dict[str, list[ReplayResult]] = {}
    for ref in meta.runs:
        lines = (run_dir / ref.file).read_text(encoding="utf-8").splitlines()
        results[ref.label] = [
            ReplayResult.model_validate_json(line) for line in lines if line.strip()
        ]
    return meta, results
