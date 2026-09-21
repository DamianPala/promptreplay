"""OpenRouter's model-page feed: the reported cache-hit share, per endpoint and day.

The feed behind `openrouter.ai/models/<slug>` is not part of OpenRouter's documented API
(`/api/v1/...`); it lives under `/api/frontend/v1/...`, needs no key, and may change shape
without notice. Three anonymous GETs build one day's figure:

1. `GET /api/v1/models` -- the documented model list, read only for `canonical_slug`: the
   dated permaslug (`z-ai/glm-5.3-flash-20260826`) the other two endpoints key their data by.
2. `GET /api/frontend/v1/stats/cache-hit-rate-comparison?permaslug=...&timeRange=1w` -- one row
   per UTC day, endpoint uuid to the percentage of its prompt tokens served from cache that day.
3. `GET /api/frontend/v1/stats/effective-pricing?permaslug=...&shape=v7&variant=standard` --
   `endpointProviderSlugs` (uuid to provider slug) and `providerSummaries[].totalTokens`
   (uuid to a weight), used to pool the endpoints of one provider the feed cannot otherwise
   tell apart (`providerSummaries[].effectiveInputPrice` is deliberately not read: it folds in
   OpenRouter's own discounts, and the comparison this module feeds must use the run's own
   listed prices, not OpenRouter's observed ones).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import httpx
from pydantic import BaseModel

from promptreplay.bench.openrouter import MODELS_URL

_CACHE_HIT_URL = "https://openrouter.ai/api/frontend/v1/stats/cache-hit-rate-comparison"
_EFFECTIVE_PRICING_URL = "https://openrouter.ai/api/frontend/v1/stats/effective-pricing"
_TIMEOUT_S = 10.0


class ReportedShare(BaseModel):
    """One provider's reported cache share on the day asked for."""

    share_pct: float
    endpoints: int
    """How many of the provider's endpoints this figure pools; `> 1` means the feed could not
    tell them apart (they share a name and a price -- OpenRouter numbers them `(1)`, `(2)`)."""
    tokens: int | None = None
    """The token weight the pooling used; `None` when no endpoint in the group had one."""


class ReportedAverage(BaseModel):
    """One day's reported shares, by provider slug, for one model."""

    day: str
    """`YYYY-MM-DD`, UTC."""
    model: str = ""
    """The spec's model (`z-ai/glm-5.3-flash`), not the dated `permaslug` below -- what
    `_apply_one` matches a spec against, so a probe pinning two models to the same provider
    never puts one model's share on the other's row. Empty only for a run whose block was
    written before this field existed: `report` then loads that run instead of failing on it,
    and shows no comparison, since nothing says which model the shares belong to."""
    permaslug: str
    fetched_at: str
    """ISO 8601 UTC, when the three requests were made."""
    shares: dict[str, ReportedShare] = {}


def canonical_slug(payload: dict[str, Any], model: str) -> str | None:
    """`model`'s dated permaslug from a `/api/v1/models` payload, or `None` when not listed."""
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    for raw in cast("list[Any]", data):
        if not isinstance(raw, dict):
            continue
        entry = cast("dict[str, Any]", raw)
        if entry.get("id") == model:
            slug = entry.get("canonical_slug")
            return slug if isinstance(slug, str) else None
    return None


def reported_average(
    hit_payload: dict[str, Any],
    pricing_payload: dict[str, Any],
    *,
    day: str,
    model: str,
    permaslug: str,
    fetched_at: str,
) -> ReportedAverage:
    """Fold the two feed payloads into one day's shares, pooled by provider slug.

    A provider absent from the day's row is absent from `shares`: no traffic that day is not
    a zero share, it is nothing to compare against. Grouping happens over every uuid the
    pricing payload names for a provider, weighted by `totalTokens` when every endpoint in
    the group has one; if even one is missing from `providerSummaries`, the whole group pools
    at equal weight instead, since one known total would otherwise drown the rest out or a
    missing one would silently score zero. A payload missing either request's shape raises
    `ValueError` naming what was missing.
    """
    day_row = _day_row(hit_payload, day)
    slugs = _provider_slugs(pricing_payload)
    tokens_by_uuid = _tokens_by_uuid(pricing_payload)
    by_provider: dict[str, list[str]] = {}
    for endpoint_id, slug in slugs.items():
        by_provider.setdefault(slug, []).append(endpoint_id)

    shares: dict[str, ReportedShare] = {}
    for slug, endpoint_ids in by_provider.items():
        present = [endpoint_id for endpoint_id in endpoint_ids if endpoint_id in day_row]
        if not present:
            continue
        shares[slug] = _pool(present, day_row, tokens_by_uuid)
    return ReportedAverage(
        day=day, model=model, permaslug=permaslug, fetched_at=fetched_at, shares=shares
    )


def _pool(
    endpoint_ids: list[str], day_row: dict[str, float], tokens_by_uuid: dict[str, int]
) -> ReportedShare:
    weights = [tokens_by_uuid.get(endpoint_id) for endpoint_id in endpoint_ids]
    if all(weight is not None for weight in weights):
        effective = [int(weight) for weight in weights if weight is not None]
        total_tokens: int | None = sum(effective)
    else:
        # One endpoint of this provider has no weight to trust, so none of them are
        # weighted: an equal share beats letting a single known total drown the rest out.
        effective = [1] * len(endpoint_ids)
        total_tokens = None
    pooled = sum(day_row[uid] * w for uid, w in zip(endpoint_ids, effective, strict=True))
    pooled /= sum(effective)
    return ReportedShare(
        share_pct=round(pooled, 1), endpoints=len(endpoint_ids), tokens=total_tokens
    )


def _day_row(hit_payload: dict[str, Any], day: str) -> dict[str, float]:
    data = hit_payload.get("data")
    if not isinstance(data, list):
        raise ValueError("cache-hit-rate-comparison payload: missing 'data'")
    rows = cast("list[Any]", data)
    row = next((entry for entry in rows if _is_day(entry, day)), None)
    if row is None:
        return {}
    values = cast("dict[str, Any]", row).get("y")
    if not isinstance(values, dict):
        raise ValueError("cache-hit-rate-comparison payload: row missing 'y'")
    pairs = cast("dict[Any, Any]", values)
    return {
        str(uuid): float(share) for uuid, share in pairs.items() if isinstance(share, int | float)
    }


def _is_day(entry: object, day: str) -> bool:
    if not isinstance(entry, dict):
        return False
    return str(cast("dict[str, Any]", entry).get("x", "")).startswith(day)


def _provider_slugs(pricing_payload: dict[str, Any]) -> dict[str, str]:
    data = pricing_payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("effective-pricing payload: missing 'data'")
    slugs = cast("dict[str, Any]", data).get("endpointProviderSlugs")
    if not isinstance(slugs, dict):
        raise ValueError("effective-pricing payload: missing 'endpointProviderSlugs'")
    pairs = cast("dict[Any, Any]", slugs)
    return {str(uuid): str(slug) for uuid, slug in pairs.items() if isinstance(slug, str)}


def _tokens_by_uuid(pricing_payload: dict[str, Any]) -> dict[str, int]:
    data = pricing_payload.get("data")
    raw_summaries = (
        cast("dict[str, Any]", data).get("providerSummaries") if isinstance(data, dict) else None
    )
    if not isinstance(raw_summaries, list):
        return {}
    tokens: dict[str, int] = {}
    for raw in cast("list[Any]", raw_summaries):
        if not isinstance(raw, dict):
            continue
        summary = cast("dict[str, Any]", raw)
        endpoint_id = summary.get("endpointId")
        total = summary.get("totalTokens")
        if isinstance(endpoint_id, str) and isinstance(total, int | float):
            tokens[endpoint_id] = int(total)
    return tokens


async def fetch_reported_average(
    client: httpx.AsyncClient, model: str, *, day: str
) -> ReportedAverage | None:
    """The model's reported average for `day`, or `None` on any failure.

    Three anonymous GETs, no `Authorization` header: the feed is undocumented and a key must
    never be sent to a route OpenRouter has not published. No retries -- the caller says once
    that the comparison is unavailable and moves on.
    """
    try:
        models_resp = await client.get(MODELS_URL, timeout=_TIMEOUT_S)
        models_resp.raise_for_status()
        permaslug = canonical_slug(models_resp.json(), model)
        if permaslug is None:
            return None
        hit_resp = await client.get(
            _CACHE_HIT_URL,
            params={"permaslug": permaslug, "timeRange": "1w"},
            timeout=_TIMEOUT_S,
        )
        hit_resp.raise_for_status()
        pricing_resp = await client.get(
            _EFFECTIVE_PRICING_URL,
            params={"permaslug": permaslug, "shape": "v7", "variant": "standard"},
            timeout=_TIMEOUT_S,
        )
        pricing_resp.raise_for_status()
        return reported_average(
            hit_resp.json(),
            pricing_resp.json(),
            day=day,
            model=model,
            permaslug=permaslug,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    except (httpx.HTTPError, ValueError):
        return None
