"""OpenRouter /generation and /models/{model}/endpoints clients."""

from __future__ import annotations

import asyncio
import re
from typing import Any, cast

import httpx
from pydantic import BaseModel

from promptreplay.bench.targets import Prices

_NORMALIZE = re.compile(r"[^a-z0-9]")

_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
_ZDR_URL = "https://openrouter.ai/api/v1/endpoints/zdr"


def normalize_provider(name: str) -> str:
    """Lowercase, keep [a-z0-9] only, so "SiliconFlow" and "siliconflow" match."""
    return _NORMALIZE.sub("", name.lower())


class Endpoint(BaseModel):
    """One endpoint of a model, as `/models/{model}/endpoints` reports it.

    `latency_ms_30m` and `throughput_30m` are the median (`p50`) of the percentile object
    OpenRouter returns for a request carrying the API key; without a key both are `null`,
    which is why a `--sort latency` sweep asks for the key.
    """

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


class ZdrEndpoint(BaseModel):
    """One entry of the Zero Data Retention list: which model, and which of its endpoints."""

    model_id: str
    tag: str


def _price_per_million(pricing: dict[str, Any], key: str) -> float:
    raw = pricing.get(key)
    if raw is None:
        return 0.0
    try:
        return float(raw) * 1e6
    except (TypeError, ValueError):
        return 0.0


def _p50(value: object) -> float | None:
    """The median of a health field: a bare number, or the `{p50, p75, p90, p99}` object.

    OpenRouter answers these fields with a plain number for an unauthenticated request and
    with the full percentile object once the key is present; the ranking reads p50 because
    that is the request a user is most likely to get.
    """
    if isinstance(value, dict):
        value = cast("dict[str, Any]", value).get("p50")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


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
                latency_ms_30m=_p50(entry.get("latency_last_30m")),
                throughput_30m=_p50(entry.get("throughput_last_30m")),
                status=entry.get("status"),
                supports_implicit_caching=entry.get("supports_implicit_caching"),
            )
        )
    return endpoints


def parse_zdr_endpoints(payload: dict[str, Any]) -> list[ZdrEndpoint]:
    """The `/endpoints/zdr` list: every ZDR endpoint of every model, flat."""
    endpoints: list[ZdrEndpoint] = []
    for raw in cast("list[object]", payload.get("data") or []):
        if not isinstance(raw, dict):
            continue
        entry = cast("dict[str, Any]", raw)
        endpoints.append(
            ZdrEndpoint(model_id=str(entry.get("model_id") or ""), tag=str(entry.get("tag") or ""))
        )
    return endpoints


async def fetch_endpoints(
    client: httpx.AsyncClient, model: str, *, api_key: str | None = None
) -> list[Endpoint]:
    """The model's endpoints; with `api_key` the response also carries the health percentiles."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    resp = await client.get(_ENDPOINTS_URL.format(model=model), headers=headers)
    resp.raise_for_status()
    return parse_endpoints(resp.json())


async def fetch_zdr_endpoints(client: httpx.AsyncClient) -> list[ZdrEndpoint]:
    """The Zero Data Retention list; it covers every model and needs no API key."""
    resp = await client.get(_ZDR_URL)
    resp.raise_for_status()
    return parse_zdr_endpoints(resp.json())


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
