"""The spec column every table shares: one caption, whole `@tags`, elision as a last resort.

A list of specs is a list of near-identical `target:model@provider` strings, and the part
that tells one row from another is the tail. Cutting the middle out of those strings is what
makes `@relace/…` and `@relace/…-fp8` read alike, so what a table needs is a decision about
the whole column at once: which shared prefix moves into a caption above the table, and what
each row then shows on its own. `column_labels` makes that decision for the probe tables and
for the estimate table, so a spec reads the same in either place.

(`targets.py` imports nothing here and this module imports nothing from the package: a
table's labels are strings, and both the estimate and the probe tables need them.)
"""

from __future__ import annotations

from collections.abc import Sequence

_ELLIPSIS = "…"

__all__ = ["column_labels", "elide", "named", "rendered"]


def column_labels(labels: Sequence[str], *, label_width: int) -> tuple[str | None, list[str]]:
    """The caption and the label column of every row, as one shared decision.

    A `target:model` that at least two rows share moves into the caption, so those rows
    show their provider tails (`@novita`, `@gmicloud`) while a row of another model keeps
    its own name. The caption may name several shared prefixes; when nothing is shared
    there is no caption and every row keeps its whole label. A shortened set that repeats
    is thrown away: two rows reading the same string name neither of them.
    """
    heads = [_head(label) for label in labels]
    shared = sorted({head for head in heads if head and heads.count(head) > 1})
    column = [
        _short_label(label, head, shared, label_width)
        for label, head in zip(labels, heads, strict=True)
    ]
    if not named(column):
        shared = []
        column = [elide(label, label_width) for label in labels]
    caption = "specs: " + ", ".join(f"{head}@<provider>" for head in shared) if shared else None
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


def _short_label(label: str, head: str, shared: Sequence[str], width: int) -> str:
    """`label` without the shared head, or whole when this spec's model is its own."""
    tail = label[len(head) :] if head in shared else ""
    return elide(tail or label, width)
