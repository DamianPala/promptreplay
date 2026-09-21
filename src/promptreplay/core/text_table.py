"""The plain fixed-width table every human renderer in the tool draws.

One function, no styling, no terminal-width awareness: every column is padded to its
widest cell and cells are separated by ` | `, so a table looks the same piped to a file as
it does on a terminal. `core` renders it directly (`config show`, the generic paged-record
table); `bench.labels` re-exports it for `bench`/`commands` callers that also need
`column_labels`' endpoint-specific shortening.
"""

from __future__ import annotations

from collections.abc import Sequence

_SEPARATOR = " | "

__all__ = ["text_table"]


def text_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A fixed-width table: the header, a rule of the same widths, then one line per row.

    Every cell is padded to the widest in its column, so a reader compares a cell by looking
    at the column rather than counting characters. A table with no rows is its header and
    the rule, which is what a listing with nothing left to show looks like.
    """
    widths = [
        max([len(columns[index]), *(len(row[index]) for row in rows)])
        for index in range(len(columns))
    ]
    lines = [_join(columns, widths), _join(["-" * width for width in widths], widths)]
    lines.extend(_join(row, widths) for row in rows)
    return lines


def _join(cells: Sequence[str], widths: Sequence[int]) -> str:
    return _SEPARATOR.join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
