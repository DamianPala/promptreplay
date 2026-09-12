"""Tests for provibench.bench.replay: prepare_body, end-to-end replay, persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from provibench.bench.replay import (
    ReplayOptions,
    ReplayResult,
    load_run,
    prepare_body,
    replay_run,
    write_run,
)
from provibench.bench.targets import Prices, RunSpec, Target
from provibench.bench.trace import TraceEntry, Usage


def _target(
    kind: Literal["openrouter", "anthropic"] = "anthropic",
    prices: dict[str, Prices] | None = None,
) -> Target:
    return Target(
        name="t",
        url="https://api.test/v1/messages",
        api_key_env="X_KEY",
        kind=kind,
        prices=prices or {},
    )


_MINIMAL_OK_BODY = {"id": "msg_1", "usage": {"input_tokens": 1, "output_tokens": 1}}


def _trace_entry(seq: int, *, extra_body: dict[str, Any] | None = None) -> TraceEntry:
    body: dict[str, Any] = {
        "model": "orig-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
    }
    body.update(extra_body or {})
    return TraceEntry(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        path="/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        body=body,
        conversation="c1",
    )


# --- prepare_body -----------------------------------------------------------


def test_prepare_body_overrides_model_stream_max_tokens_leaves_rest() -> None:
    spec = RunSpec(target=_target(), model="new-model")
    opts = ReplayOptions(max_tokens=5)
    body = {
        "model": "orig",
        "stream": True,
        "max_tokens": 1024,
        "messages": [],
        "system": "sys",
        "tools": [{"a": 1}],
        "metadata": {"user_id": "u1"},
    }
    prepared = prepare_body(body, spec, opts)
    assert prepared["model"] == "new-model"
    assert prepared["stream"] is False
    assert prepared["max_tokens"] == 5
    assert prepared["system"] == "sys"
    assert prepared["tools"] == [{"a": 1}]
    assert prepared["metadata"] == {"user_id": "u1"}
    assert "provider" not in prepared
    assert body["model"] == "orig"  # original untouched


def test_prepare_body_injects_provider_when_pinned() -> None:
    spec = RunSpec(target=_target("openrouter"), model="m", providers=["novita", "siliconflow"])
    prepared = prepare_body({"messages": []}, spec, ReplayOptions())
    assert prepared["provider"] == {"only": ["novita", "siliconflow"], "allow_fallbacks": False}


def test_prepare_body_no_provider_key_when_unpinned() -> None:
    spec = RunSpec(target=_target("openrouter"), model="m")
    prepared = prepare_body({"messages": []}, spec, ReplayOptions())
    assert "provider" not in prepared


def test_prepare_body_strips_thinking_blocks_from_assistant_messages() -> None:
    spec = RunSpec(target=_target(), model="m")
    opts = ReplayOptions(strip_thinking=True)
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": "hello"},
                ],
            },
        ]
    }
    prepared = prepare_body(body, spec, opts)
    assert prepared["messages"][1]["content"] == [{"type": "text", "text": "hello"}]
    assert prepared["messages"][0]["content"] == "hi"


def test_prepare_body_strip_thinking_placeholder_when_message_left_empty() -> None:
    spec = RunSpec(target=_target(), model="m")
    opts = ReplayOptions(strip_thinking=True)
    body = {
        "messages": [
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]},
        ]
    }
    prepared = prepare_body(body, spec, opts)
    assert prepared["messages"][0]["content"] == [{"type": "text", "text": " "}]


def test_prepare_body_without_strip_thinking_leaves_blocks_untouched() -> None:
    spec = RunSpec(target=_target(), model="m")
    body = {"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]}]}
    prepared = prepare_body(body, spec, ReplayOptions(strip_thinking=False))
    assert prepared["messages"][0]["content"] == [{"type": "thinking", "thinking": "x"}]


# --- replay_run end-to-end ---------------------------------------------------


def test_replay_run_openrouter_end_to_end_enriches_from_generation() -> None:
    target = Target(
        name="openrouter",
        url="https://openrouter.test/v1/messages",
        api_key_env="OR",
        kind="openrouter",
    )
    spec = RunSpec(target=target, model="deepseek/model", providers=["novita"])
    entries = [_trace_entry(1), _trace_entry(2)]
    opts = ReplayOptions(max_tokens=1)

    endpoints_payload = {
        "data": {
            "endpoints": [
                {
                    "provider_name": "Novita",
                    "tag": "novita",
                    "pricing": {
                        "prompt": "0.0000001",
                        "completion": "0.0000002",
                        "input_cache_read": "0.00000001",
                        "input_cache_write": "0.0000001",
                    },
                }
            ]
        }
    }
    message_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=endpoints_payload)
        if request.url.path.endswith("/generation"):
            gen_id = request.url.params.get("id")
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": gen_id,
                        "provider_name": "Novita",
                        "native_tokens_prompt": 100,
                        "native_tokens_cached": 40,
                        "native_tokens_completion": 1,
                        "total_cost": 0.001,
                        "cache_discount": 0.5,
                    }
                },
            )
        if request.url.path.endswith("/v1/messages"):
            message_calls["n"] += 1
            return httpx.Response(
                200,
                json={
                    "id": f"msg_{message_calls['n']}",
                    "model": "deepseek/model",
                    "provider": "Novita",
                    "usage": {
                        "input_tokens": 60,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 40,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
        return httpx.Response(404)

    async def run() -> list[ReplayResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_run(spec, entries, opts, "fake-key", client=client)

    results = asyncio.run(run())
    assert len(results) == 2
    for result in results:
        assert result.status == 200
        assert result.provider == "Novita"
        assert result.prompt_total == 100  # overridden by generation lookup
        assert result.cached == 40
        assert result.output_tokens == 1
        assert result.billed_total == 0.001
        assert result.cache_discount == 0.5
        assert result.cost is not None
        assert result.cost.source == "openrouter-endpoint"
        assert result.cost.total == pytest.approx((100 - 40) * 0.1e-6 + 40 * 0.01e-6 + 1 * 0.2e-6)


def test_replay_run_anthropic_end_to_end_uses_price_table() -> None:
    prices = {"model-a": Prices(input=1, cache_read=0.1, cache_write=1, output=2)}
    spec = RunSpec(target=_target("anthropic", prices), model="model-a")
    entries = [_trace_entry(1)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "model-a",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    async def run() -> list[ReplayResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_run(spec, entries, ReplayOptions(), "fake-key", client=client)

    results = asyncio.run(run())
    assert len(results) == 1
    result = results[0]
    assert result.status == 200
    assert result.cost is not None
    assert result.cost.source == "table"
    assert result.cost.total == pytest.approx(100 * 1e-6 * 1 + 10 * 1e-6 * 2)


def test_replay_run_anthropic_missing_price_table_notes_and_no_cost() -> None:
    spec = RunSpec(target=_target("anthropic"), model="model-a")
    entries = [_trace_entry(1)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_MINIMAL_OK_BODY)

    async def run() -> list[ReplayResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_run(spec, entries, ReplayOptions(), "fake-key", client=client)

    results = asyncio.run(run())
    assert results[0].cost is None
    assert results[0].note == "no price table for model-a"


def test_replay_run_retries_once_after_400_thinking_error() -> None:
    prices = {"model-a": Prices(input=1, cache_read=1, cache_write=1, output=1)}
    spec = RunSpec(target=_target("anthropic", prices), model="model-a")
    entry = _trace_entry(1, extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}})

    bodies_seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies_seen.append(body)
        if "thinking" in body:
            message = "max_tokens must be greater than thinking.budget_tokens"
            return httpx.Response(400, json={"error": {"message": message}})
        return httpx.Response(200, json=_MINIMAL_OK_BODY)

    async def run() -> list[ReplayResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_run(spec, [entry], ReplayOptions(), "fake-key", client=client)

    results = asyncio.run(run())
    assert len(bodies_seen) == 2
    assert "thinking" in bodies_seen[0]
    assert "thinking" not in bodies_seen[1]
    assert results[0].status == 200
    assert results[0].note == "thinking param dropped"


def test_replay_run_400_without_trigger_words_does_not_retry() -> None:
    spec = RunSpec(target=_target("anthropic"), model="model-a")
    entry = _trace_entry(1)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "invalid api key"}})

    async def run() -> list[ReplayResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_run(spec, [entry], ReplayOptions(), "fake-key", client=client)

    results = asyncio.run(run())
    assert calls["n"] == 1
    assert results[0].status == 400
    assert results[0].note is None


def test_replay_run_network_error_becomes_status_zero_and_does_not_abort() -> None:
    spec = RunSpec(target=_target("anthropic"), model="model-a")
    entries = [_trace_entry(1), _trace_entry(2)]

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    async def run() -> list[ReplayResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await replay_run(spec, entries, ReplayOptions(), "fake-key", client=client)

    results = asyncio.run(run())
    assert len(results) == 2
    assert all(r.status == 0 for r in results)
    assert all("boom" in (r.error or "") for r in results)


# --- persistence --------------------------------------------------------------


def test_write_run_and_load_run_round_trip(tmp_path: Path) -> None:
    spec = RunSpec(target=_target("anthropic"), model="model-a")
    opts = ReplayOptions(max_tokens=3)
    result = ReplayResult(
        seq=1,
        turn=1,
        status=200,
        latency_ms=12.3,
        message_id="msg_1",
        model="model-a",
        usage=Usage(input_tokens=1, output_tokens=1),
        prompt_total=1,
        cached=0,
        cache_write=0,
        output_tokens=1,
    )
    results = {spec.label: [result]}

    run_dir = write_run(tmp_path, "trace1", "conv1", [spec], opts, results=results)
    meta, loaded = load_run(run_dir)

    assert meta.trace == "trace1"
    assert meta.conversation == "conv1"
    assert meta.options.max_tokens == 3
    assert meta.runs[0].label == spec.label
    assert meta.runs[0].file == f"{spec.slug}.jsonl"
    assert loaded[spec.label][0].message_id == "msg_1"
    assert loaded[spec.label][0].status == 200
