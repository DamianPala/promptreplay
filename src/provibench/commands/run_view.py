"""The `history` and `compare` documents: one row per spec and run, and the deltas of two.

`history` and `compare` read the same rows — one spec in one run — so the document shape
is declared once and built once. `commands.run_tables` renders these documents; the
documents are the source of truth, so a script and a reader see the same numbers.

The numbers stay what the runs measured: a hit rate is a fraction and an effective price
is USD per 1M prompt tokens, in the units a script subtracts. A table prints the fraction
as a per cent because that is how two of them are compared at a glance, and the delta it
shows is the difference of the displayed values in points for hit rate, and in document
units for the other metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from provibench.core.documents import (
    Document,
    JsonSchema,
    array,
    integer,
    nullable_number,
    nullable_string,
    obj,
    string,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from provibench.bench.run_delta import Comparison, MetricPair
    from provibench.bench.run_history import RunNumbers, RunRow, Series


def _all(properties: dict[str, JsonSchema]) -> JsonSchema:
    """A document schema whose every property is required: these documents always carry them."""
    return obj(properties, required=list(properties))


_RUN_ROW = _all(
    {
        "trace": string(),
        "created": string(),
        "run_dir": string(),
        "protocol": string(),
        "spec": string(),
        "target": string(),
        "model": string(),
        "provider": string(),
        "hit_rate": nullable_number(),
        "eff_per_m_prompt": nullable_number(),
        "ttft_ms": nullable_number(),
        "gen_tok_s": nullable_number(),
        "errors": integer(),
        "listed_input": nullable_number(),
        "listed_cache_read": nullable_number(),
        "listed_source": string(),
        "quantization": nullable_string(),
        "context_length": nullable_number(),
        "uptime_1d": nullable_number(),
        "status": nullable_string(),
    }
)
_SERIES = _all(
    {
        "spec": string(),
        "provider": string(),
        "trace": string(),
        "protocol": string(),
        "runs": integer(),
        "hit_rates": array(nullable_number()),
    }
)
HISTORY_OUTPUT = _all({"runs": integer(), "rows": array(_RUN_ROW), "series": array(_SERIES)})

_RUN_REF: dict[str, JsonSchema] = {
    "run_dir": string(),
    "trace": string(),
    "created": string(),
    "protocol": string(),
}
_METRIC = _all({"a": nullable_number(), "b": nullable_number(), "delta": nullable_number()})
_SPEC_DELTA = _all(
    {
        "spec": string(),
        "hit_rate": _METRIC,
        "eff_per_m_prompt": _METRIC,
        "ttft_ms": _METRIC,
        "gen_tok_s": _METRIC,
    }
)
_PRICE_DELTA = _all({"spec": string(), "price_in": _METRIC, "price_cache_read": _METRIC})
COMPARE_OUTPUT = _all(
    {
        "run_a": obj(_RUN_REF, required=list(_RUN_REF)),
        "run_b": obj(_RUN_REF, required=list(_RUN_REF)),
        "rows": array(_SPEC_DELTA),
        "listed": array(_PRICE_DELTA),
        "only_in_a": array(string()),
        "only_in_b": array(string()),
    }
)


def history_document(runs: Sequence[RunNumbers], series: Sequence[Series]) -> Document:
    """The `history` success document: one row per spec and run, plus the series.

    The rows come out in the order the table reads them — one provider at a time, oldest
    first — and the series in the same order, so a sparkline line sits under the rows it
    belongs to. `--json` carries that order too rather than the directory order.
    """
    from provibench.bench.run_history import rows_in_order

    return {
        "runs": len(runs),
        "rows": [row_document(row) for row in rows_in_order(runs)],
        "series": [series_document(entry) for entry in series],
    }


def compare_document(comparison: Comparison) -> Document:
    """The `compare` success document: the pair names, the deltas and the unpaired specs."""
    return {
        "run_a": run_ref_document(comparison.a),
        "run_b": run_ref_document(comparison.b),
        "rows": [
            {
                "spec": row.spec,
                "hit_rate": metric_document(row.hit_rate),
                "eff_per_m_prompt": metric_document(row.eff_per_m_prompt),
                "ttft_ms": metric_document(row.ttft_ms),
                "gen_tok_s": metric_document(row.gen_tok_s),
            }
            for row in comparison.rows
        ],
        "listed": [
            {
                "spec": row.spec,
                "price_in": metric_document(row.price_in),
                "price_cache_read": metric_document(row.price_cache_read),
            }
            for row in comparison.listed
        ],
        "only_in_a": list(comparison.only_in_a),
        "only_in_b": list(comparison.only_in_b),
    }


def metric_document(metric: MetricPair) -> Document:
    """One metric as `{a, b, delta}`; the delta is `null` when a side has no number."""
    return {"a": metric.a, "b": metric.b, "delta": metric.delta}


def run_ref_document(run: RunNumbers) -> Document:
    """Which run this is: its directory, trace, stamp and protocol."""
    return {
        "run_dir": str(run.run_dir),
        "trace": run.trace,
        "created": run.created,
        "protocol": run.protocol,
    }


def row_document(row: RunRow) -> Document:
    """One row of the history table, with the snapshot facts the table leaves to `--json`.

    Quantization, context length, uptime and status are not table columns — the table is
    about what the runs measured — but they are exactly what the snapshot exists for, so
    a script watching a provider's drift reads them here.
    """
    return {
        "trace": row.trace,
        "created": row.created,
        "run_dir": str(row.run_dir),
        "protocol": row.protocol,
        "spec": row.spec,
        "target": row.target,
        "model": row.model,
        "provider": row.provider,
        "hit_rate": row.hit_rate,
        "eff_per_m_prompt": row.eff_per_m_prompt,
        "ttft_ms": row.ttft_ms,
        "gen_tok_s": row.gen_tok_s,
        "errors": row.errors,
        "listed_input": row.listed_input,
        "listed_cache_read": row.listed_cache_read,
        "listed_source": row.listed_source,
        "quantization": row.quantization,
        "context_length": row.context_length,
        "uptime_1d": row.uptime_1d,
        "status": row.status,
    }


def series_document(series: Series) -> Document:
    """One spec, trace and protocol's hit rate per run, oldest first."""
    return {
        "spec": series.spec,
        "provider": series.provider,
        "trace": series.trace,
        "protocol": series.protocol,
        "runs": series.runs,
        "hit_rates": list(series.hit_rates),
    }
