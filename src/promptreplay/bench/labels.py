"""The endpoint column every table shares: one caption, whole `@tags`, elision as a last resort.

A list of endpoints is a list of near-identical `target:model@provider` strings, and the part
that tells one row from another is the tail. Cutting the middle out of those strings is what
makes `@relace/…` and `@relace/…-fp8` read alike, so what a table needs is a decision about
the whole column at once: which shared prefix moves into a caption above the table, and what
each row then shows on its own. `column_labels` makes that decision for the probe tables and
for the estimate table, so an endpoint reads the same in either place.

`text_table` -- the columns-and-rows layout every table in the tool draws -- lives in
`core.text_table` instead: `core/config_command.py` and `core/render.py` need it too, and
`core` may not import from here. It is re-exported below for every caller that already reads
it from this module.

(`targets.py` imports nothing here and this module imports nothing else from the package: a
table's labels are strings, and every table in the tool needs them.)
"""

from __future__ import annotations

from collections.abc import Sequence

from promptreplay.core.text_table import text_table

_ELLIPSIS = "…"

__all__ = [
    "column_labels",
    "elide",
    "join_and",
    "named",
    "rendered",
    "size_label",
    "text_table",
]


def size_label(tokens: int) -> str:
    """A prompt size as the reader sees it: `20k`, or the bare token count below 1k."""
    return f"{tokens / 1000:.0f}k" if tokens >= 1000 else str(tokens)


def join_and(names: Sequence[str]) -> str:
    """`A`, `A and B`, or `A, B and C`: a plain list, no Oxford comma past two items."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def column_labels(labels: Sequence[str], *, label_width: int) -> tuple[str | None, list[str]]:
    """The caption and the label column of every row, as one shared decision.

    A `target:model` that at least two rows share moves into the caption, so those rows
    show their provider tails (`@novita`, `@gmicloud`) while a row of another model keeps
    its own name. A single row does the same when its own label does not fit `label_width`:
    a one-endpoint table still reads `@sail-research/fp8` rather than an elided full label.
    The caption may name several shared prefixes; when nothing is shared there is no caption
    and every row keeps its whole label. A shortened set that repeats is thrown away: two
    rows reading the same string name neither of them.
    """
    heads = [_head(label) for label in labels]
    shared = _shared_heads(labels, label_width=label_width)
    column = [
        _short_label(label, head, shared, label_width)
        for label, head in zip(labels, heads, strict=True)
    ]
    if not named(column):
        shared = []
        column = [elide(label, label_width) for label in labels]
    caption = "endpoints: " + ", ".join(f"{head}@<provider>" for head in shared) if shared else None
    return caption, column


def elide(cell: str, width: int, *, keep_end: bool = True) -> str:
    """`cell` cut to `width` with the middle dropped: `openrouter:deepseek/…@novita`.

    A label's provider suffix is its shortest distinguishing part, so it survives whole and
    the head keeps what is left. A label without one keeps both of its ends instead: its
    model sits at the end and its target at the start, and cutting either loses the pair of
    names that tell one row from another — so a `target:model` keeps its target whole and
    spends the rest on the tail of the model. `keep_end=False` is for a cell whose start is
    what matters — a comma-separated list of drift markers — and cuts its tail.
    """
    if len(cell) <= width:
        return cell
    if not keep_end:
        return f"{cell[: width - len(_ELLIPSIS)]}{_ELLIPSIS}"
    at = cell.rfind("@")
    if at > 0:
        tail = cell[at:]
        keep = width - len(tail) - len(_ELLIPSIS)
        if keep >= 1:
            return f"{cell[:keep]}{_ELLIPSIS}{tail}"
    colon = cell.find(":")
    if 0 < colon <= width - 2 - len(_ELLIPSIS):
        # `target:model` is read as a pair. A row the caption does not cover has to keep
        # its target whole: elided from both ends it reads as one more endpoint of the
        # model the caption names, which is the one thing this column exists to say.
        keep = width - colon - 1 - len(_ELLIPSIS)
        return f"{cell[: colon + 1]}{_ELLIPSIS}{cell[-keep:]}"
    head = max(1, (width - len(_ELLIPSIS)) // 3)
    return f"{cell[:head]}{_ELLIPSIS}{cell[-(width - len(_ELLIPSIS) - head) :]}"


def named(labels: Sequence[str]) -> bool:
    """Whether every row still has a name of its own after the cut."""
    return len(set(labels)) == len(labels)


def rendered(labels: Sequence[str]) -> int:
    """The width a label column really takes: what the rows show, not what was planned."""
    return max((len(label) for label in labels), default=0)


def _head(label: str) -> str:
    """A label's `target:model` part, without its `@provider` suffix."""
    head, at, _ = label.rpartition("@")
    return head if at else label


def _shared_heads(labels: Sequence[str], *, label_width: int | None = None) -> list[str]:
    """The `target:model` heads that move into the caption: shared by at least two rows, or,
    when there is exactly one row and `label_width` is given, that row's own head if the
    whole label does not fit it. `label_width=None` (the default, used where no row's own
    width is at hand) never folds a lone row, which is what `column_labels` opts into and
    every other caller keeps.
    """
    heads = [_head(label) for label in labels]
    shared = sorted({head for head in heads if head and heads.count(head) > 1})
    if shared or label_width is None or len(labels) != 1:
        return shared
    [label], [head] = labels, heads
    # `head == label` means the label has no `@provider` to split off (a native endpoint):
    # folding it would name a caption's `@<provider>` next to a row that repeats the whole
    # label anyway, which explains nothing a plain elided row does not already say.
    if not head or head == label or len(label) <= label_width:
        return []
    return [head]


def _short_label(label: str, head: str, shared: Sequence[str], width: int) -> str:
    """`label` without the shared head, or whole when this endpoint's model is its own."""
    tail = label[len(head) :] if head in shared else ""
    return elide(tail or label, width)
