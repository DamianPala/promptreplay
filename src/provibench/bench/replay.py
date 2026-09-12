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

from provibench.bench.openrouter import (
    Generation,
    fetch_endpoints,
    fetch_generation,
    normalize_provider,
)
from provibench.bench.pricing import CostBreakdown, compute_cost
from provibench.bench.sse import ParsedMessage, parse_json_message
from provibench.bench.targets import Prices, RunSpec, resolve_api_key
from provibench.bench.trace import TraceEntry, Usage

_STRIPPED_BLOCK_TYPES = {"thinking", "redacted_thinking"}
_THINKING_RETRY_TRIGGERS = ("max_tokens", "budget", "thinking")
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class ReplayOptions(BaseModel):
    max_tokens: int = 1
    delay_s: float = 0.0
    strip_thinking: bool = False
    timeout_s: float = 300.0


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


def _build_headers(entry: TraceEntry, api_key: str) -> dict[str, str]:
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "anthropic-version": entry.headers.get("anthropic-version", _DEFAULT_ANTHROPIC_VERSION),
    }
    beta = entry.headers.get("anthropic-beta")
    if beta:
        headers["anthropic-beta"] = beta
    return headers


def _should_retry_without_thinking(body_text: str) -> bool:
    lowered = body_text.lower()
    return any(trigger in lowered for trigger in _THINKING_RETRY_TRIGGERS)


def _parse_body(resp: httpx.Response) -> ParsedMessage:
    try:
        data = resp.json()
    except ValueError:
        return ParsedMessage(error=resp.text[:500])
    if not isinstance(data, dict):
        return ParsedMessage(error=str(data)[:500])
    return parse_json_message(cast("dict[str, Any]", data))


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
    return await client.post(spec.target.url, headers=headers, json=body, timeout=opts.timeout_s)


async def _replay_turn(
    entry: TraceEntry,
    turn: int,
    spec: RunSpec,
    opts: ReplayOptions,
    api_key: str,
    *,
    client: httpx.AsyncClient,
) -> ReplayResult:
    prepared = prepare_body(entry.body, spec, opts)
    headers = _build_headers(entry, api_key)
    note: str | None = None
    started = perf_counter()

    try:
        resp = await _post(client, spec, headers, prepared, opts)
    except httpx.HTTPError as exc:
        elapsed = (perf_counter() - started) * 1000
        return _empty_result(entry, turn, spec, elapsed, error=str(exc), note=None)

    if resp.status_code == 400 and _should_retry_without_thinking(resp.text):
        prepared.pop("thinking", None)
        note = "thinking param dropped"
        try:
            resp = await _post(client, spec, headers, prepared, opts)
        except httpx.HTTPError as exc:
            elapsed = (perf_counter() - started) * 1000
            return _empty_result(entry, turn, spec, elapsed, error=str(exc), note=note)

    latency_ms = (perf_counter() - started) * 1000
    parsed = _parse_body(resp)
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


def _append_note(existing: str | None, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def _apply_generation(
    result: ReplayResult, gen: Generation | None, price_by_key: dict[str, Prices]
) -> None:
    if gen is None:
        result.note = _append_note(result.note, "generation lookup failed")
        return

    result.generation = gen.raw
    result.billed_total = gen.total_cost
    result.cache_discount = gen.cache_discount
    if gen.provider_name:
        result.provider = gen.provider_name
    if gen.native_tokens_prompt is not None:
        result.prompt_total = gen.native_tokens_prompt
    if gen.native_tokens_cached is not None:
        result.cached = gen.native_tokens_cached
    if gen.native_tokens_completion is not None:
        result.output_tokens = gen.native_tokens_completion

    if result.cached > result.prompt_total:
        result.note = _append_note(result.note, "cached exceeds prompt_total; input cost skipped")
        return

    prices = price_by_key.get(normalize_provider(gen.provider_name or ""))
    if prices is None:
        note = f"no endpoint price match for provider {gen.provider_name!r}"
        result.note = _append_note(result.note, note)
        return

    result.cost = compute_cost(
        input_tokens=result.prompt_total - result.cached,
        cache_read=result.cached,
        cache_write=result.cache_write,
        output_tokens=result.output_tokens,
        prices=prices,
        source="openrouter-endpoint",
    )


async def _enrich_openrouter(
    spec: RunSpec, results: list[ReplayResult], client: httpx.AsyncClient, api_key: str
) -> None:
    endpoints = await fetch_endpoints(client, spec.model)
    price_by_key: dict[str, Prices] = {}
    for ep in endpoints:
        price_by_key[normalize_provider(ep.tag)] = ep.prices
        price_by_key[normalize_provider(ep.provider_name)] = ep.prices

    sem = asyncio.Semaphore(4)
    successful = [r for r in results if r.status == 200 and r.message_id]

    async def _fetch(result: ReplayResult) -> None:
        message_id = result.message_id
        if message_id is None:
            return  # unreachable: `successful` already filtered on message_id
        async with sem:
            gen = await fetch_generation(client, api_key, message_id)
        _apply_generation(result, gen, price_by_key)

    await asyncio.gather(*(_fetch(r) for r in successful))


def _enrich_anthropic(spec: RunSpec, results: list[ReplayResult]) -> None:
    prices = spec.target.prices.get(spec.model)
    for result in results:
        if result.status != 200:
            continue
        if prices is None:
            result.note = _append_note(result.note, f"no price table for {spec.model}")
            continue
        result.cost = compute_cost(
            input_tokens=result.usage.input_tokens,
            cache_read=result.cached,
            cache_write=result.cache_write,
            output_tokens=result.output_tokens,
            prices=prices,
            source="table",
        )


async def replay_run(
    spec: RunSpec,
    entries: list[TraceEntry],
    opts: ReplayOptions,
    api_key: str,
    *,
    client: httpx.AsyncClient,
    on_progress: Callable[[ReplayResult], None] | None = None,
) -> list[ReplayResult]:
    results: list[ReplayResult] = []
    for turn, entry in enumerate(entries, start=1):
        result = await _replay_turn(entry, turn, spec, opts, api_key, client=client)
        results.append(result)
        if on_progress is not None:
            on_progress(result)
        if opts.delay_s and turn < len(entries):
            await asyncio.sleep(opts.delay_s)

    if spec.target.kind == "openrouter":
        await _enrich_openrouter(spec, results, client, api_key)
    else:
        _enrich_anthropic(spec, results)
    return results


async def replay_all(
    specs: list[RunSpec],
    entries: list[TraceEntry],
    opts: ReplayOptions,
    env: Mapping[str, str],
    *,
    on_progress: Callable[[ReplayResult], None] | None = None,
) -> dict[str, list[ReplayResult]]:
    # Resolve keys up front so a missing key fails before any request is sent.
    api_keys = {spec.label: resolve_api_key(spec.target, env) for spec in specs}
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
                    on_progress=on_progress,
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


class RunMeta(BaseModel):
    trace: str
    conversation: str
    created: str
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
        trace=trace_name, conversation=conversation, created=created, options=opts, runs=refs
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
