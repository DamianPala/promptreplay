"""The JSON shapes for `RunSummary` and `ProbeSummary`, and the sweep block's own schema.

Split out of `summary_view` purely for its line budget: the schemas and the functions that
reshape a summary into one are the part of that module with nothing to do with rendering.

`RunSummary.providers_seen` is a string-keyed map, which the O4 schema subset cannot
describe (it has no `additionalProperties`); `summary_to_document` turns it into a
`[{provider, count}]` list so the whole summary fits a fixed-property schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from promptreplay.core.documents import (
    Document,
    JsonSchema,
    array,
    boolean,
    integer,
    nullable_boolean,
    nullable_integer,
    nullable_number,
    nullable_object,
    nullable_string,
    number,
    obj,
    string,
)

if TYPE_CHECKING:
    from promptreplay.bench.openrouter_stats import ReportedAverage
    from promptreplay.bench.pricing import CostBreakdown
    from promptreplay.bench.probe_summary import ProbeSummary
    from promptreplay.bench.summary import RunSummary


def _all(properties: dict[str, JsonSchema]) -> JsonSchema:
    """A document schema whose every property is required: these documents always carry them."""
    return obj(properties, required=list(properties))


_PROVIDER_COUNT = obj({"provider": string(), "count": integer()}, required=["provider", "count"])
_COST_PROPERTIES: dict[str, JsonSchema] = {
    "input": number(),
    "cache_read": number(),
    "cache_write": number(),
    "output": number(),
    "total": number(),
    "source": string(),
}
SUMMARY = _all(
    {
        "label": string(),
        "requested_providers": array(string()),
        "turns": integer(),
        "ok": integer(),
        "errors": integer(),
        "providers_seen": array(_PROVIDER_COUNT),
        "drift": integer(),
        "prompt_total": integer(),
        "cached_total": integer(),
        "cache_write_total": integer(),
        "output_total": integer(),
        "hit_ratio": nullable_number(),
        "curve": array(number()),
        "cost": nullable_object(_COST_PROPERTIES, required=list(_COST_PROPERTIES)),
        "billed_total": nullable_number(),
        "effective_per_m_prompt": nullable_number(),
        "latency_p50_ms": nullable_number(),
        "latency_p95_ms": nullable_number(),
        "notes": array(string()),
    }
)

_TTL_READ = _all(
    {
        "offset_s": integer(),
        "hit": boolean(),
        "fraction": nullable_number(),
        "offset_actual_s": nullable_number(),
    }
)
_RUNG_SUMMARY = _all(
    {
        "endpoint": string(),
        "rung": integer(),
        "prompt_cold": integer(),
        "cached_cold": integer(),
        "hit_rate": number(),
        "cached_fraction": nullable_number(),
        "hits": array(nullable_number()),
        "cold_ms": number(),
        "warm_ms": nullable_number(),
        "ttft_ms": nullable_number(),
        "gen_tok_s": nullable_number(),
        "fingerprint": nullable_string(),
        "ttl": array(_TTL_READ),
        "cache_write_cold": integer(),
        "errors": integer(),
        "rate_limited": integer(),
        "retries": integer(),
        "skipped": boolean(),
        "cold_error": nullable_string(),
    }
)
PROBE_SUMMARY = _all(
    {
        "label": string(),
        "rungs": array(_RUNG_SUMMARY),
        "hit_rate": nullable_number(),
        "cached_fraction": nullable_number(),
        "h": nullable_number(),
        "first_hit_rate": nullable_number(),
        "first_cached_fraction": nullable_number(),
        "first_h": nullable_number(),
        "input_price": nullable_number(),
        "cache_read_price": nullable_number(),
        "price_source": string(),
        "priced_as": string(),
        "listed_input": nullable_number(),
        "listed_cache_read": nullable_number(),
        "listed_source": string(),
        "eff_per_m_prompt": nullable_number(),
        "billed_usd": nullable_number(),
        "spend_usd": nullable_number(),
        "session_prompt_usd": nullable_number(),
        "output_tokens": integer(),
        "output_usd": nullable_number(),
        "providers_seen": array(_PROVIDER_COUNT),
        "served": string(),
        "model_seen": nullable_string(),
        "models_seen": array(string()),
        "tokens_delta_pct": nullable_number(),
        "fingerprint_match": nullable_boolean(),
        "ttft_ms": nullable_number(),
        "gen_tok_s": nullable_number(),
        "reference": boolean(),
        "drift": nullable_string(),
        "errors": integer(),
        "rate_limited": integer(),
        "retries": integer(),
        "skipped": integer(),
        "notes": array(string()),
        "or_avg_share_pct": nullable_number(),
        "or_avg_pooled": boolean(),
        "or_avg_eff_per_m_prompt": nullable_number(),
        "or_avg_session_prompt_usd": nullable_number(),
        "vs_or_avg_pct": nullable_number(),
    }
)

_REPORTED_SHARE_PROPERTIES: dict[str, JsonSchema] = {
    "share_pct": number(),
    "endpoints": integer(),
    "tokens": nullable_integer(),
}
"""`ReportedAverage.shares`'s value shape; the object itself is keyed by provider slug, which
the O4 schema subset (no `additionalProperties`) cannot enumerate, so its own schema below
declares it a bare object rather than listing every possible slug as a property."""
REPORTED_AVERAGE_PROPERTIES: dict[str, JsonSchema] = {
    "day": string(),
    "model": string(),
    "permaslug": string(),
    "fetched_at": string(),
    "shares": {"type": "object"},
}
REPORTED_AVERAGE = nullable_object(
    REPORTED_AVERAGE_PROPERTIES, required=list(REPORTED_AVERAGE_PROPERTIES)
)

_SWEEP_DROP = _all(
    {"tag": string(), "endpoint": string(), "reason": string(), "checked": boolean()}
)
_SWEEP_RANKED = _all(
    {
        "tag": string(),
        "price_input": number(),
        "uptime_1d": nullable_number(),
        "status": nullable_string(),
        "latency_p50_ms": nullable_number(),
        "throughput_p50_tok_s": nullable_number(),
    }
)
SWEEP_BLOCK_PROPERTIES: dict[str, JsonSchema] = {
    "model": string(),
    "target": string(),
    "included": array(string()),
    "excluded": array(string()),
    "quantization": array(string()),
    "pinned": array(string()),
    "sort": string(),
    "top": nullable_integer(),
    "zdr": boolean(),
    "uptime_floor": number(),
    "check": boolean(),
    "dropped": array(_SWEEP_DROP),
    "ranking": array(_SWEEP_RANKED),
}
"""The `sweep` block's fields: `sweep` publishes them, `report` re-reads them as nullable."""
SWEEP_BLOCK = obj(SWEEP_BLOCK_PROPERTIES, required=list(SWEEP_BLOCK_PROPERTIES))


def summary_to_document(summary: RunSummary) -> Document:
    """`summary` reshaped so every field has a fixed set of JSON properties."""
    return {
        "label": summary.label,
        "requested_providers": list(summary.requested_providers),
        "turns": summary.turns,
        "ok": summary.ok,
        "errors": summary.errors,
        "providers_seen": [
            {"provider": name, "count": count}
            for name, count in sorted(summary.providers_seen.items())
        ],
        "drift": summary.drift,
        "prompt_total": summary.prompt_total,
        "cached_total": summary.cached_total,
        "cache_write_total": summary.cache_write_total,
        "output_total": summary.output_total,
        "hit_ratio": summary.hit_ratio,
        "curve": list(summary.curve),
        "cost": _cost_document(summary.cost),
        "billed_total": summary.billed_total,
        "effective_per_m_prompt": summary.effective_per_m_prompt,
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
        "notes": list(summary.notes),
    }


def probe_summary_to_document(summary: ProbeSummary) -> Document:
    """`summary` reshaped so `providers_seen` has a fixed set of JSON properties."""
    return {
        "label": summary.label,
        "rungs": [rung.model_dump() for rung in summary.rungs],
        "hit_rate": summary.hit_rate,
        "cached_fraction": summary.cached_fraction,
        "h": summary.h,
        "first_hit_rate": summary.first_hit_rate,
        "first_cached_fraction": summary.first_cached_fraction,
        "first_h": summary.first_h,
        "input_price": summary.input_price,
        "cache_read_price": summary.cache_read_price,
        "price_source": summary.price_source,
        "priced_as": summary.priced_as,
        "listed_input": summary.listed_input,
        "listed_cache_read": summary.listed_cache_read,
        "listed_source": summary.listed_source,
        "eff_per_m_prompt": summary.eff_per_m_prompt,
        "billed_usd": summary.billed_usd,
        "spend_usd": summary.spend_usd,
        "session_prompt_usd": summary.session_prompt_usd,
        "output_tokens": summary.output_tokens,
        "output_usd": summary.output_usd,
        "providers_seen": [
            {"provider": name, "count": count}
            for name, count in sorted(summary.providers_seen.items())
        ],
        "served": summary.served,
        "model_seen": summary.model_seen,
        "models_seen": list(summary.models_seen),
        "tokens_delta_pct": summary.tokens_delta_pct,
        "fingerprint_match": summary.fingerprint_match,
        "ttft_ms": summary.ttft_ms,
        "gen_tok_s": summary.gen_tok_s,
        "reference": summary.reference,
        "drift": summary.drift,
        "errors": summary.errors,
        "rate_limited": summary.rate_limited,
        "retries": summary.retries,
        "skipped": summary.skipped,
        "notes": list(summary.notes),
        "or_avg_share_pct": summary.or_avg_share_pct,
        "or_avg_pooled": summary.or_avg_pooled,
        "or_avg_eff_per_m_prompt": summary.or_avg_eff_per_m_prompt,
        "or_avg_session_prompt_usd": summary.or_avg_session_prompt_usd,
        "vs_or_avg_pct": summary.vs_or_avg_pct,
    }


def reported_average_document(average: ReportedAverage | None) -> Document | None:
    """A `ReportedAverage` (or `None`) as the document `probe`/`report` publish it."""
    if average is None:
        return None
    return {
        "day": average.day,
        "model": average.model,
        "permaslug": average.permaslug,
        "fetched_at": average.fetched_at,
        "shares": {slug: share.model_dump() for slug, share in average.shares.items()},
    }


def _cost_document(cost: CostBreakdown | None) -> Document | None:
    if cost is None:
        return None
    return {
        "input": cost.input,
        "cache_read": cost.cache_read,
        "cache_write": cost.cache_write,
        "output": cost.output,
        "total": cost.total,
        "source": cost.source,
    }
