"""OpenRouter /generation and /models/{model}/endpoints clients."""

from __future__ import annotations

import asyncio
import re
from typing import Any, cast

import httpx
from pydantic import BaseModel

from provibench.bench.targets import Prices

_NORMALIZE = re.compile(r"[^a-z0-9]")

_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
_GENERATION_URL = "https://openrouter.ai/api/v1/generation"


def normalize_provider(name: str) -> str:
    """Lowercase, keep [a-z0-9] only, so "SiliconFlow" and "siliconflow" match."""
    return _NORMALIZE.sub("", name.lower())


class Endpoint(BaseModel):
    provider_name: str
    tag: str
    quantization: str | None = None
    context_length: int | None = None
    prices: Prices
    uptime_30m: float | None = None
    uptime_1d: float | None = None
    latency_ms_30m: float | None = None
    throughput_30m: float | None = None
    status: int | str | None = None
    supports_implicit_caching: bool | None = None


def _price_per_million(pricing: dict[str, Any], key: str) -> float:
    raw = pricing.get(key)
    if raw is None:
        return 0.0
    try:
        return float(raw) * 1e6
    except (TypeError, ValueError):
        return 0.0


def parse_endpoints(payload: dict[str, Any]) -> list[Endpoint]:
    entries: list[dict[str, Any]] = []
    data = payload.get("data")
    if isinstance(data, dict):
        raw_entries = cast("dict[str, Any]", data).get("endpoints")
        if isinstance(raw_entries, list):
            entries = cast("list[dict[str, Any]]", raw_entries)

    endpoints: list[Endpoint] = []
    for entry in entries:
        pricing: dict[str, Any] = entry.get("pricing") or {}
        prices = Prices(
            input=_price_per_million(pricing, "prompt"),
            cache_read=_price_per_million(pricing, "input_cache_read"),
            cache_write=_price_per_million(pricing, "input_cache_write"),
            output=_price_per_million(pricing, "completion"),
        )
        endpoints.append(
            Endpoint(
                provider_name=entry.get("provider_name") or "",
                tag=entry.get("tag") or "",
                quantization=entry.get("quantization"),
                context_length=entry.get("context_length"),
                prices=prices,
                uptime_30m=entry.get("uptime_last_30m"),
                uptime_1d=entry.get("uptime_last_1d"),
                latency_ms_30m=entry.get("latency_last_30m"),
                throughput_30m=entry.get("throughput_last_30m"),
                status=entry.get("status"),
                supports_implicit_caching=entry.get("supports_implicit_caching"),
            )
        )
    return endpoints


async def fetch_endpoints(client: httpx.AsyncClient, model: str) -> list[Endpoint]:
    resp = await client.get(_ENDPOINTS_URL.format(model=model))
    resp.raise_for_status()
    return parse_endpoints(resp.json())


class Generation(BaseModel):
    id: str
    provider_name: str | None = None
    model: str | None = None
    native_tokens_prompt: int | None = None
    native_tokens_cached: int | None = None
    native_tokens_completion: int | None = None
    native_tokens_reasoning: int | None = None
    total_cost: float | None = None
    cache_discount: float | None = None
    latency_ms: float | None = None
    generation_time_ms: float | None = None
    raw: dict[str, Any]


def _parse_generation(data: dict[str, Any]) -> Generation:
    return Generation(
        id=data.get("id", ""),
        provider_name=data.get("provider_name"),
        model=data.get("model"),
        native_tokens_prompt=data.get("native_tokens_prompt"),
        native_tokens_cached=data.get("native_tokens_cached"),
        native_tokens_completion=data.get("native_tokens_completion"),
        native_tokens_reasoning=data.get("native_tokens_reasoning"),
        total_cost=data.get("total_cost"),
        cache_discount=data.get("cache_discount"),
        latency_ms=data.get("latency"),
        generation_time_ms=data.get("generation_time"),
        raw=data,
    )


async def fetch_generation(
    client: httpx.AsyncClient,
    api_key: str,
    gen_id: str,
    *,
    attempts: int = 8,
    backoff_s: float = 1.5,
) -> Generation | None:
    """Fetch a generation; OpenRouter's index lags recording by a few seconds."""
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, attempts + 1):
        resp = await client.get(_GENERATION_URL, params={"id": gen_id}, headers=headers)
        if resp.status_code == 404:
            if attempt == attempts:
                return None
            await asyncio.sleep(backoff_s * attempt)
            continue
        resp.raise_for_status()
        return _parse_generation(resp.json().get("data") or {})
    return None
