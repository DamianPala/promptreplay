"""Tests for promptreplay.bench.openrouter."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from promptreplay.bench.openrouter import (
    Generation,
    fetch_generation,
    normalize_provider,
    parse_endpoints,
)


def test_parse_endpoints_realistic_payload() -> None:
    payload: dict[str, Any] = {
        "data": {
            "endpoints": [
                {
                    "provider_name": "SiliconFlow",
                    "tag": "siliconflow",
                    "quantization": "fp8",
                    "context_length": 128000,
                    "pricing": {
                        "prompt": "0.00000015",
                        "completion": "0.0000006",
                        "input_cache_read": "0.000000003",
                        "input_cache_write": "0.00000015",
                    },
                    "uptime_last_30m": 99.9,
                    "uptime_last_1d": 99.5,
                    "latency_last_30m": 450.0,
                    "throughput_last_30m": 80.0,
                    "status": 0,
                    "supports_implicit_caching": True,
                },
                {
                    "provider_name": "Novita",
                    "tag": "novita",
                    "pricing": {},
                },
            ]
        }
    }
    endpoints = parse_endpoints(payload)
    assert len(endpoints) == 2

    first = endpoints[0]
    assert first.provider_name == "SiliconFlow"
    assert first.tag == "siliconflow"
    assert first.prices.input == pytest.approx(0.15)
    assert first.prices.output == pytest.approx(0.6)
    assert first.prices.cache_read == pytest.approx(0.003)
    assert first.prices.cache_write == pytest.approx(0.15)
    assert first.supports_implicit_caching is True

    second = endpoints[1]
    assert second.prices.input == 0.0
    assert second.quantization is None
    assert second.context_length is None


def test_parse_endpoints_missing_endpoints_key() -> None:
    assert parse_endpoints({"data": {}}) == []
    assert parse_endpoints({}) == []


def test_normalize_provider() -> None:
    assert normalize_provider("SiliconFlow") == "siliconflow"
    assert normalize_provider("Open-Router_2") == "openrouter2"


def test_fetch_generation_retries_on_404_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={"data": {"id": "gen-1", "provider_name": "Novita", "total_cost": 0.01}},
        )

    transport = httpx.MockTransport(handler)

    async def run() -> Generation | None:
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_generation(client, "key", "gen-1", attempts=5, backoff_s=0.001)

    gen = asyncio.run(run())
    assert gen is not None
    assert gen.id == "gen-1"
    assert gen.provider_name == "Novita"
    assert calls["n"] == 3


def test_fetch_generation_gives_up_after_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    async def run() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_generation(client, "key", "gen-1", attempts=2, backoff_s=0.001)

    assert asyncio.run(run()) is None


def test_fetch_generation_raises_on_other_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    transport = httpx.MockTransport(handler)

    async def run() -> object:
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_generation(client, "key", "gen-1")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())
