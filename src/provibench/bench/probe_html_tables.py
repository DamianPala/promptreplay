"""How the probe's two tables are reshaped for the HTML page: money, relabelling, dropped
columns, and the `hits` cell -- none of it touching the shared `TableBlock` that
`probe_tables.probe_blocks` hands to the text and Markdown renderers too.

Kept apart from `html_report` because this is the part with decisions in it: which columns
count as constant enough to drop, what counts as a "full" cache hit, and what the four
hardest column names mean to someone who has not opened the README. The output is still
`TableBlock` data, not markup, so it renders through the same table function the rest of
the page uses.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.probe_answer import join_and, trace_money
from provibench.bench.probe_summary import ProbeSummary, RungSummary
from provibench.bench.probe_tables import TableBlock

__all__ = [
    "BLANKS_LINE",
    "ENDPOINT_HELP",
    "GLOSSARY",
    "RUNG_HELP",
    "dropped_line",
    "endpoint_view",
    "rung_view",
]

_HIT_FULL = 0.98
"""Mirrors `probe_tables._HIT_FULL`: the cached fraction at or above which a read counts as
a full cache hit for the `hits` cell's `N/M cached` count."""

_CONSTANT_PLACEHOLDERS = {"-", "0"}
"""A column showing only one of these across every row teaches a reader nothing (item 6)."""

GLOSSARY = (
    "Glossary: <strong>rung</strong> — a recorded turn, probed at three prompt sizes; "
    "<strong>cached %</strong> (share of the prompt served from the cache) — 100% means "
    "the whole prompt was already cached; <strong>eff $/M</strong> — the real price per "
    "1M prompt tokens at the measured hit rate; <strong>1st hit %</strong> — the share of "
    "rungs whose first warm read after the write hit the cache: the cold-start case, and "
    "a gap below hit % means the cache needs a few requests before it helps."
)

BLANKS_LINE = "<code>-</code> means not measured; the caveats say why."

ENDPOINT_HELP: dict[str, str] = {
    "endpoint": "The target, model and pinned provider this row measured.",
    "hit %": "Warm reads that found any part of the prefix, divided by warm reads served.",
    "1st hit %": (
        "The same fraction, counting only each rung's first warm read after the write: the "
        "cold-start case (a fresh session, a restart, a gateway switching providers)."
    ),
    "cached %": (
        "On a hit, the share of the prompt served from the cache; 100% means the whole prompt."
    ),
    "eff $/M": (
        "Hit-weighted prompt price per 1M tokens: (1 - h) x input + h x cache read, with h "
        "over every warm read, the steady state of a long session."
    ),
    "in $/M": "The listed input price used for eff $/M; repriced when the endpoint is unpinned.",
    "cold ms (rung 1)": "The first rung's cold prefill latency, in milliseconds.",
    "warm ms (rung 1)": "The first rung's warm prefill latency, in milliseconds.",
    "TTFT ms": "median across rungs",
    "tok/s": "median across rungs",
    "errors": "Requests without a usable answer; a failed cold write is not a cache miss.",
    "drift": "Provider, model or token-count drift versus the reference endpoint; - means none.",
    "this trace $": "What this endpoint would bill for a session shaped like this trace.",
}

RUNG_HELP: dict[str, str] = {
    "endpoint": ENDPOINT_HELP["endpoint"],
    "rung": "A recorded turn, probed at the smallest, middle and largest size.",
    "prompt": "The cold request's prompt size, in tokens.",
    "cached cold": "What the cold write read back; a non-zero value indicates contamination.",
    "hits": "How many of this rung's warm reads came back fully cached.",
    "cold ms": "This rung's cold prefill latency, in milliseconds.",
    "warm ms": "This rung's warm prefill latency, in milliseconds.",
    "TTFT ms": "Time to the first streamed token, for this rung.",
    "tok/s": "Output tokens per second, for this rung.",
    "errors": ENDPOINT_HELP["errors"],
    "ttl": "One cell per --ttl offset: whether the cache was still there after that wait.",
}


def endpoint_view(
    block: TableBlock, summaries: Sequence[ProbeSummary]
) -> tuple[TableBlock, list[str]]:
    """The per-provider table as the HTML page shows it: relabelled, with its money column,
    constant columns dropped -- none of it touching the shared table the text report reads."""
    columns = (*_relabel_first_rung(block.columns), "this trace $")
    rows = tuple(
        (*row, trace_money(summary.session_prompt_usd))
        for row, summary in zip(block.rows, summaries, strict=True)
    )
    return _drop_constant_columns(TableBlock(columns=columns, rows=rows))


def _relabel_first_rung(columns: Sequence[str]) -> tuple[str, ...]:
    """`cold ms`/`warm ms` are the first rung only (item 7); the rung table says as much by
    being per-rung, but the per-provider row needs the reminder in its own header."""
    mapping = {"cold ms": "cold ms (rung 1)", "warm ms": "warm ms (rung 1)"}
    return tuple(mapping.get(column, column) for column in columns)


def rung_view(
    block: TableBlock, summaries: Sequence[ProbeSummary]
) -> tuple[TableBlock, list[str], dict[tuple[int, int], str]]:
    """The rung table with constant columns dropped and its `hits` cell made readable."""
    dropped_block, dropped = _drop_constant_columns(block)
    reformatted, titles = _reformat_hits(dropped_block, summaries)
    return reformatted, dropped, titles


def _drop_constant_columns(block: TableBlock) -> tuple[TableBlock, list[str]]:
    """Drop a column (never the label column) whose every cell is the same placeholder.

    `drift` reading `-` in every row of a probe with no drift, or `cached cold` reading `0`
    in every row of a run with no contamination, spend width teaching the reader nothing.
    """
    if not block.rows or len(block.columns) <= 1:
        return block, []
    keep = [0]
    dropped: list[str] = []
    for index in range(1, len(block.columns)):
        values = {row[index] for row in block.rows}
        if len(values) == 1 and next(iter(values)) in _CONSTANT_PLACEHOLDERS:
            dropped.append(block.columns[index])
        else:
            keep.append(index)
    if not dropped:
        return block, []
    columns = tuple(block.columns[index] for index in keep)
    rows = tuple(tuple(row[index] for index in keep) for row in block.rows)
    return TableBlock(columns=columns, rows=rows), dropped


def dropped_line(dropped: Sequence[str]) -> str:
    return f"{join_and(dropped)} omitted: no values"


def _reformat_hits(
    block: TableBlock, summaries: Sequence[ProbeSummary]
) -> tuple[TableBlock, dict[tuple[int, int], str]]:
    """`0.81 1 1 1 1 1` becomes `5/6 cached`, with the sequence kept in the cell's `title`.

    A failed read is not a miss and is not cached either: it is counted into the total and
    named separately (`4/6 cached, 2 failed`), so an error is never read as a clean miss.
    """
    if "hits" not in block.columns:
        return block, {}
    column = block.columns.index("hits")
    flat_rungs: list[RungSummary] = [rung for summary in summaries for rung in summary.rungs]
    rows = list(block.rows)
    titles: dict[tuple[int, int], str] = {}
    for index, (row, rung) in enumerate(zip(rows, flat_rungs, strict=True)):
        original = row[column]
        rows[index] = (*row[:column], _hits_cell(rung.hits, original), *row[column + 1 :])
        if original not in ("-", ""):
            titles[index, column] = original
    return TableBlock(columns=block.columns, rows=tuple(rows)), titles


def _hits_cell(hits: Sequence[float | None], original: str) -> str:
    total = len(hits)
    if not total:
        return original
    cached = sum(1 for hit in hits if hit is not None and hit >= _HIT_FULL)
    failed = sum(1 for hit in hits if hit is None)
    text = f"{cached}/{total} cached"
    return f"{text}, {failed} failed" if failed else text
