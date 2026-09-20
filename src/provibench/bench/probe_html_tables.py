"""How the probe's two tables are reshaped for the HTML page: money, relabelling, dropped
columns, the `hits` cell and the drift cell -- none of it touching the shared `TableBlock`
that `probe_tables.probe_blocks` hands to the text and Markdown renderers too.

Kept apart from `html_report` because this is the part with decisions in it: which columns
count as constant enough to drop, what counts as a "full" cache hit, and what the hardest
column names mean to someone who has not opened the README. The output is still `TableBlock`
data or plain strings, not markup, so it renders through the same table function the rest of
the page uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from provibench.bench.labels import size_label
from provibench.bench.probe_summary import ProbeSummary, RungSummary
from provibench.bench.probe_tables import TableBlock

__all__ = [
    "ENDPOINT_CAPTION_HTML",
    "GROUPS",
    "endpoint_caption_html",
    "endpoint_help",
    "endpoint_view",
    "group_header_row",
    "rung_help",
    "rung_view",
    "trace_money",
]

_HIT_FULL = 0.98
"""Mirrors `probe_tables._HIT_FULL`: the cached fraction at or above which a read counts as
a full cache hit, rather than a partial one, in the `hits` cell's words (item 7)."""

_CONSTANT_PLACEHOLDERS = {"-", "0", ""}
"""A column showing only one of these across every row teaches a reader nothing (item 6)."""

ENDPOINT_CAPTION_HTML = (
    "Cheapest by <code>eff $/M</code> first. A cache miss pays the input price, a hit pays "
    "the cache price; <code>eff $/M</code> is what each endpoint costs at the hit rate this "
    "run measured."
)
"""The endpoint table's caption (item 1): a fact about the table, not a verdict on one row,
so it survives however the run's own numbers turn out. It names the column it sorts by,
because the first row can carry the highest `in $/M` and "cheapest" alone reads as false."""


def endpoint_caption_html(labels: Sequence[str], model: str, *, folded: bool) -> str:
    """The table's caption, plus what the folded `@tag` labels stand for when they are folded.

    The text report keeps its `endpoints: <shared prefix>` line above the table; the page
    says the same thing here, in the caption, as a sentence -- a spec string nobody typed,
    printed under the title, is not a fact a reader can do anything with.
    """
    pinned = [label for label in labels if label.startswith("@")]
    if not folded or not pinned:
        return ENDPOINT_CAPTION_HTML
    subject = "row goes" if len(pinned) == 1 else "rows go"
    serve = "it serves" if len(pinned) == 1 else "all serve"
    return (
        f"{ENDPOINT_CAPTION_HTML} The {_count_word(len(pinned))} @ {subject} through "
        f"OpenRouter, pinned to the named provider; {serve} {escape(model)}."
    )


_ENDPOINT_TOOLTIP = (
    "Who served the requests: the API, the model and, for OpenRouter rows, the pinned provider."
)
_ERRORS_TOOLTIP = "Requests without a usable answer. A failed cold write is not a cache miss."

_NUMBER_WORDS: dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

GROUPS: dict[str, tuple[str, ...]] = {
    "cache": ("hit %", "1st hit %", "cached %"),
    "price": ("in $/M", "cache $/M", "eff $/M", "this trace $"),
}
"""Which named group a column belongs to; `cold ms`/`warm ms` carry a size suffix that
varies per run, so `_group_of` matches them by prefix instead of by exact name."""


def trace_money(value: float | None) -> str:
    """A trace-shaped dollar amount as the reader thinks in it: `$0.0060`, not `$/M`.

    Four decimals, always: this is the same `session_prompt_usd` the text report already
    prints through `commands.probe_spend`, so the two renderers would otherwise disagree
    over one number. A fixed count is also what a column is read down -- trimming zeros put
    `$0.006` and `$0.0056` in neighbouring rows, which has to be compared digit by digit --
    and at three decimals a cheap session rounds to `$0.000`, which reads as free.
    """
    return "-" if value is None else f"${value:.4f}"


def _count_word(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


def _trace_size_words(tokens: int) -> str:
    """A trace's total prompt tokens as the tooltip states it: `1.8M`, not `1804855`."""
    if tokens >= 1_000_000:
        return f"{tokens / 1e6:.1f}M"
    return f"{round(tokens / 1000)}k"


def _reference_cold_size(summaries: Sequence[ProbeSummary]) -> int:
    """The reference endpoint's first rung prompt size, or `0` when none measured one."""
    ranked = sorted(summaries, key=lambda summary: not summary.reference)
    for summary in ranked:
        if summary.rungs and summary.rungs[0].prompt_cold > 0:
            return summary.rungs[0].prompt_cold
    return 0


def _cold_warm_labels(size: int) -> tuple[str, str]:
    """The endpoint table's `cold ms`/`warm ms` headers, sized when a reference size is known."""
    if not size:
        return "cold ms", "warm ms"
    label = size_label(size)
    return f"cold ms · {label}", f"warm ms · {label}"


def endpoint_help(summaries: Sequence[ProbeSummary], trace_prompt_tokens: object) -> dict[str, str]:
    """Every endpoint-table tooltip, built for this run's own turn count and trace size."""
    turns = len({rung.rung for summary in summaries for rung in summary.rungs})
    turns_word = _count_word(turns) if turns else "each"
    size = _reference_cold_size(summaries)
    cold_label, warm_label = _cold_warm_labels(size)
    cold_text = (
        f"Time to answer the first, uncached request at the {size_label(size)}-token turn."
        if size
        else "Time to answer the first, uncached request at this endpoint's smallest turn."
    )
    trace_text = (
        f"Prompt bill for a session like the recorded one "
        f"({_trace_size_words(trace_prompt_tokens)} prompt tokens) at this endpoint's eff $/M."
        if isinstance(trace_prompt_tokens, int) and trace_prompt_tokens > 0
        else "Prompt bill for a session like the recorded one at this endpoint's eff $/M."
    )
    return {
        "endpoint": _ENDPOINT_TOOLTIP,
        "hit %": "Of all repeat requests, the share the cache answered at least in part.",
        "1st hit %": (
            "Of the first repeat after each cache write, the share the cache answered: what a "
            "fresh session or a restart sees. A gap below hit % means the cache needs a few "
            "requests before it helps."
        ),
        "cached %": (
            "When the cache hit, the share of the prompt it covered. 100 means the whole prompt."
        ),
        "in $/M": "Listed price per 1M input tokens: what a cache miss costs.",
        "cache $/M": "Listed price per 1M cached prompt tokens: what a cache hit costs.",
        "eff $/M": (
            "Price per 1M prompt tokens at the measured hit rate: misses pay the input "
            "price, hits the cache price, weighted over every repeat request."
        ),
        cold_label: cold_text,
        warm_label: (
            "Time to answer the same request once its prompt was cached. The gap to cold "
            "ms is what the cache saves in latency."
        ),
        "TTFT ms": (
            f"Time to first token on a separate streamed request, median over the "
            f"{turns_word} turns."
        ),
        "tok/s": (
            f"Output tokens per second on that streamed request, median over the "
            f"{turns_word} turns."
        ),
        "errors": _ERRORS_TOOLTIP,
        "drift": (
            "Whether the endpoint answered as expected: same provider as pinned, same "
            "model, same token count as the reference row. Empty means yes."
        ),
        "this trace $": trace_text,
    }


def rung_help() -> dict[str, str]:
    """Every per-turn-table tooltip: static, since each row is read against its own turn."""
    return {
        "endpoint": _ENDPOINT_TOOLTIP,
        "turn": "One turn of the recorded session, with its prompt size.",
        "prompt": "Prompt size of the uncached request, in tokens.",
        "cached cold": "What the cold write read back. A non-zero value indicates contamination.",
        "hits": (
            "Of this turn's repeat requests, how many the cache answered. A partial hit is "
            "one where only part of the prompt was cached."
        ),
        "cold ms": "Time to answer this turn's first, uncached request.",
        "warm ms": "Time to answer this turn's repeat request, once its prompt was cached.",
        "TTFT ms": "Time to first token on this turn's streamed request.",
        "tok/s": "Output tokens per second on this turn's streamed request.",
        "errors": _ERRORS_TOOLTIP,
        "ttl": "One cell per --ttl offset: whether the cache was still there after that wait.",
    }


def _group_of(column: str) -> str | None:
    """Which group header a column sits under; `speed` matches `cold ms`/`warm ms` by
    prefix because their header carries a run-specific size suffix."""
    for name, members in GROUPS.items():
        if column in members:
            return name
    if (
        column.startswith("cold ms")
        or column.startswith("warm ms")
        or column
        in (
            "TTFT ms",
            "tok/s",
        )
    ):
        return "speed"
    return None


def group_header_row(columns: Sequence[str]) -> str:
    """The row of group headers above the endpoint table's own header (item 3)."""
    cells: list[str] = []
    index = 0
    while index < len(columns):
        group = _group_of(columns[index])
        span = 1
        while index + span < len(columns) and _group_of(columns[index + span]) == group:
            span += 1
        if group is None:
            cells.append("<th></th>" if span == 1 else f'<th colspan="{span}"></th>')
        else:
            cells.append(f'<th colspan="{span}" class="group-start">{escape(group)}</th>')
        index += span
    return f'<tr class="groups">{"".join(cells)}</tr>'


def endpoint_view(
    block: TableBlock, summaries: Sequence[ProbeSummary]
) -> tuple[TableBlock, list[str]]:
    """The per-provider table as the HTML page shows it: relabelled, with its money column
    next to the other prices, its drift cell in words -- none of it touching the shared
    table the text report reads."""
    size = _reference_cold_size(summaries)
    cold_label, warm_label = _cold_warm_labels(size)
    relabel = {"cold ms": cold_label, "warm ms": warm_label}
    columns = tuple(relabel.get(column, column) for column in block.columns)
    insert_at = columns.index("eff $/M") + 1
    columns = (*columns[:insert_at], "this trace $", *columns[insert_at:])
    rows = tuple(
        _endpoint_row(row, summary, insert_at)
        for row, summary in zip(block.rows, summaries, strict=True)
    )
    return _drop_constant_columns(TableBlock(columns=columns, rows=rows))


def _endpoint_row(row: Sequence[str], summary: ProbeSummary, insert_at: int) -> tuple[str, ...]:
    """`row` with `this trace $` inserted right after `eff $/M` and drift spelled out.

    Drift is always the table's last column (`probe_tables._RUN_COLUMNS`), so the cell being
    replaced is always `row[-1]` regardless of how many columns sit between the two.
    """
    before = row[:insert_at]
    after = (*row[insert_at:-1], _drift_cell(summary))
    return (*before, trace_money(summary.session_prompt_usd), *after)


def _drift_cell(summary: ProbeSummary) -> str:
    """`-` is reserved for "not measured" (item 5), so a clean row's cell is simply empty,
    and a `provider` marker becomes the name a reader can act on rather than a category."""
    if not summary.drift:
        return ""
    served = summary.served if summary.served and summary.served != "-" else "another provider"
    words = [
        f"served by {served}" if marker == "provider" else marker
        for marker in summary.drift.split(",")
    ]
    return ", ".join(words)


def rung_view(
    block: TableBlock, summaries: Sequence[ProbeSummary]
) -> tuple[TableBlock, list[str], dict[tuple[int, int], str]]:
    """The rung table with constant columns dropped, its `rung` column read as a turn size,
    and its `hits` cell made readable (item 7, item 8)."""
    dropped_block, dropped = _drop_constant_columns(block)
    turned = _relabel_turn(dropped_block, summaries)
    reformatted, titles = _reformat_hits(turned, summaries)
    return reformatted, dropped, titles


def _drop_constant_columns(block: TableBlock) -> tuple[TableBlock, list[str]]:
    """Drop a column (never the label column) whose every cell is the same placeholder.

    `drift` reading empty in every row of a probe with no drift, or `cached cold` reading
    `0` in every row of a run with no contamination, teaches the reader nothing; the run
    that dropped it is not announced (item 6) -- the column is simply not there.
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


def _relabel_turn(block: TableBlock, summaries: Sequence[ProbeSummary]) -> TableBlock:
    """`rung` becomes `turn`, and its cell carries the turn's own prompt size (item 8)."""
    if "rung" not in block.columns:
        return block
    column = block.columns.index("rung")
    flat_rungs = [rung for summary in summaries for rung in summary.rungs]
    columns = (*block.columns[:column], "turn", *block.columns[column + 1 :])
    rows = tuple(
        (*row[:column], _turn_cell(rung), *row[column + 1 :])
        for row, rung in zip(block.rows, flat_rungs, strict=True)
    )
    return TableBlock(columns=columns, rows=rows)


def _turn_cell(rung: RungSummary) -> str:
    if rung.prompt_cold <= 0:
        return str(rung.rung)
    return f"{rung.rung} ({size_label(rung.prompt_cold)})"


def _reformat_hits(
    block: TableBlock, summaries: Sequence[ProbeSummary]
) -> tuple[TableBlock, dict[tuple[int, int], str]]:
    """`0.93 1 1 1 1 1` becomes `6/6 hit, 1 partial`, with the reads spelled out in the
    cell's `title` -- the same count `hit %` and the cache-share chart's hover already use
    (item 7), so the three never disagree over one endpoint's reads again."""
    if "hits" not in block.columns:
        return block, {}
    column = block.columns.index("hits")
    flat_rungs = [rung for summary in summaries for rung in summary.rungs]
    rows = list(block.rows)
    titles: dict[tuple[int, int], str] = {}
    for index, (row, rung) in enumerate(zip(rows, flat_rungs, strict=True)):
        text, title = _hits_cell(rung.hits)
        rows[index] = (*row[:column], text, *row[column + 1 :])
        if title:
            titles[index, column] = title
    return TableBlock(columns=block.columns, rows=tuple(rows)), titles


def _hits_cell(hits: Sequence[float | None]) -> tuple[str, str | None]:
    total = len(hits)
    if not total:
        return "-", None
    hit_count = sum(1 for hit in hits if hit is not None and hit > 0)
    partial = sum(1 for hit in hits if hit is not None and 0 < hit < _HIT_FULL)
    failed = sum(1 for hit in hits if hit is None)
    text = f"{hit_count}/{total} hit"
    maybe_extra = (
        f"{partial} partial" if partial else None,
        f"{failed} failed" if failed else None,
    )
    extra = [part for part in maybe_extra if part]
    if extra:
        text += ", " + ", ".join(extra)
    title = " · ".join(_read_word(hit) for hit in hits)
    return text, title


def _read_word(hit: float | None) -> str:
    if hit is None:
        return "failed"
    if hit <= 0:
        return "miss"
    if hit >= _HIT_FULL:
        return "hit"
    return f"hit, {hit * 100:.0f}% cached"
