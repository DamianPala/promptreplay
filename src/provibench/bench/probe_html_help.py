"""Every tooltip the HTML page's endpoint and per-turn tables show on hover.

Kept apart from `probe_html_tables` purely for its line budget: the two tooltip
dictionaries and the sentences they build from a run's own turn count and trace size were
the largest single piece of that module. Neither function needs anything `probe_html_tables`
does not already export back to it (`count_word`, `reference_cold_size`, `cold_warm_labels`),
so there is no import to route around.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.labels import size_label
from provibench.bench.probe_html_tables import cold_warm_labels, count_word, reference_cold_size
from provibench.bench.probe_summary import ProbeSummary

__all__ = ["endpoint_help", "rung_help"]

_ENDPOINT_TOOLTIP = (
    "Who served the requests: the API, the model and, for OpenRouter rows, the pinned provider."
)
_ERRORS_TOOLTIP = "Requests without a usable answer. A failed cold write is not a cache miss."


def _trace_size_words(tokens: int) -> str:
    """A trace's total prompt tokens as the tooltip states it: `1.8M`, not `1804855`."""
    if tokens >= 1_000_000:
        return f"{tokens / 1e6:.1f}M"
    return f"{round(tokens / 1000)}k"


def endpoint_help(summaries: Sequence[ProbeSummary], trace_prompt_tokens: object) -> dict[str, str]:
    """Every endpoint-table tooltip, built for this run's own turn count and trace size.

    Every $/M here is an input (prompt) token price; output tokens are billed apart and
    never enter one of these numbers, which is why `eff $/M` and `this trace $` both say so
    themselves rather than leaving it to the caption alone (item 1).
    """
    turns = len({rung.rung for summary in summaries for rung in summary.rungs})
    turns_word = count_word(turns) if turns else "each"
    size = reference_cold_size(summaries)
    cold_label, warm_label = cold_warm_labels(size)
    cold_text = (
        f"Time to answer the first, uncached request at the {size_label(size)}-token turn."
        if size
        else "Time to answer the first, uncached request at this endpoint's smallest turn."
    )
    trace_text = (
        f"Input-token bill for a session like the recorded one "
        f"({_trace_size_words(trace_prompt_tokens)} prompt tokens) at this endpoint's eff "
        "$/M. Output tokens are not in it."
        if isinstance(trace_prompt_tokens, int) and trace_prompt_tokens > 0
        else (
            "Input-token bill for a session like the recorded one at this endpoint's eff "
            "$/M. Output tokens are not in it."
        )
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
        "in $/M": "Listed price per 1M input (prompt) tokens: what a cache miss costs.",
        "cache $/M": "Listed price per 1M cached input (prompt) tokens: what a cache hit costs.",
        "eff $/M": (
            "Input price per 1M prompt tokens at the measured hit rate: misses pay the "
            "listed input price, hits the cache price, weighted over every repeat request. "
            "Output tokens are not in it."
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
