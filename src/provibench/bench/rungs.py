"""Which turns a probe samples, and how many warm reads each one gets.

A rung is a 1-based turn `k` of the conversation; the cold request sends turn `k` and the
warm reads send turn `k+1`, so a rung always needs a following turn. Rung 1 is the
cheapest and carries the routing statistic — misses followed by hits on the same prefix are
load-balanced replicas without prefix-aware routing — so it takes the most repeats.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.trace import TraceEntry


def parse_int_list(raw: str) -> list[int]:
    """A non-empty comma-separated list of integers, as `--rungs` and `--repeats` take it."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"expected a comma-separated list of integers, got {raw!r}")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"expected a comma-separated list of integers, got {raw!r}") from exc


def broadcast_repeats(rungs: Sequence[int], repeats: Sequence[int]) -> list[int]:
    """One repeat count per rung; a short `repeats` list repeats its last value."""
    if not repeats:
        raise ValueError("--repeats must contain at least one value")
    return [repeats[min(index, len(repeats) - 1)] for index in range(len(rungs))]


def validate_rungs(rungs: Sequence[int], conversation: Sequence[TraceEntry]) -> None:
    """Every rung must point at a turn `k` that has a following turn `k+1`."""
    if not conversation:
        raise ValueError("this trace has no conversations")
    total = len(conversation)
    for rung in rungs:
        if rung < 1:
            raise ValueError(f"rung must be at least 1, got {rung}")
        if rung + 1 > total:
            raise ValueError(
                f"rung {rung} needs turn {rung + 1}, but the conversation has {total} turn(s)"
            )


def select_rungs(conversation: Sequence[TraceEntry]) -> list[int]:
    """The default rungs: the smallest, middle and largest turn that has a next turn.

    Sizes come from the recording, so every rung is a prompt the operator really sent; a
    trace with one or two turns collapses to what exists, deduplicated.
    """
    candidates = list(range(1, len(conversation)))
    if not candidates:
        raise ValueError("this trace has no turn with a following turn to probe")
    ordered = sorted(candidates, key=lambda turn: (prompt_size(conversation[turn - 1]), turn))
    return sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1]})


def prompt_size(entry: TraceEntry) -> int:
    """The turn's prompt size: its recorded usage, or the raw body when unrecorded."""
    if entry.response is None:
        return entry.body_bytes
    return entry.response.usage.prompt_total
