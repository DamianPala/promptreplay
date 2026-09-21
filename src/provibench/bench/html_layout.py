"""Generic HTML building blocks shared by the probe page and the full-replay page.

`html_report` decides what goes on a page and in what order; this module only knows how to
draw one headed table, one chart figure and the run's own identity (its timestamp and its
path relative to `runs/`) -- markup with no opinion about probe or replay. Kept apart so
`html_report`'s own line budget is spent on page assembly, not on table and chart plumbing
neither format's structure cares about.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html import escape

from provibench.bench.probe_tables import TableBlock
from provibench.core.documents import Document

__all__ = [
    "chart",
    "created",
    "created_no_seconds",
    "footer",
    "legend",
    "notes",
    "relative_run",
    "table_section",
    "theme_switch",
]

_STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$")


def created(document: Document) -> str:
    """The run's timestamp, spaced out when it is the `run.json` stamp."""
    stamp_value = str(document.get("created"))
    stamp = _STAMP.match(stamp_value)
    if stamp is None:
        return stamp_value
    date, time = stamp.group(1, 2, 3), stamp.group(4, 5, 6)
    return f"{date[0]}-{date[1]}-{date[2]} {time[0]}:{time[1]}:{time[2]}"


def created_no_seconds(document: Document) -> str:
    """The run's timestamp to the minute: a report is read later, not to the second."""
    stamp_value = str(document.get("created"))
    stamp = _STAMP.match(stamp_value)
    if stamp is None:
        return stamp_value
    date, time = stamp.group(1, 2, 3), stamp.group(4, 5, 6)
    return f"{date[0]}-{date[1]}-{date[2]} {time[0]}:{time[1]}"


def relative_run(run_dir: object) -> str:
    """The run as `runs/<trace>/<timestamp>`: where it is, without the operator's paths.

    A report is meant to be shared, and an absolute path names the machine it was made on.
    The last two segments are the run's identity — the trace and its timestamp — and the
    `runs/` prefix says what they are relative to.
    """
    segments = [part for part in str(run_dir).replace("\\", "/").split("/") if part]
    return "/".join(["runs", *segments[-2:]])


def table_section(
    title: str | None,
    block: TableBlock,
    *,
    caption_html: str | None = None,
    help: Mapping[str, str] | None = None,
    cell_titles: Mapping[tuple[int, int], str] | None = None,
    cell_extra: Mapping[tuple[int, int], str] | None = None,
    group_row: str | None = None,
    key_column: str | None = None,
) -> str:
    """One headed table, with an optional caption between the heading and the table itself.

    `title=None` skips the `<h3>`: the per-turn table sits inside a `<details>` whose own
    `<summary>` is already its heading. `key_column` names the column the table is sorted
    by; its header and cells carry `class="key"` so the stylesheet can pick it out.
    """
    heading = f"<h3>{escape(title)}</h3>" if title else ""
    cap = f'<p class="caption">{caption_html}</p>' if caption_html else ""
    body = _table(
        block,
        help=help,
        cell_titles=cell_titles,
        cell_extra=cell_extra,
        group_row=group_row,
        key_column=key_column,
    )
    return f'<section class="table">{heading}{cap}{body}</section>'


def _table(
    block: TableBlock,
    *,
    help: Mapping[str, str] | None = None,
    cell_titles: Mapping[tuple[int, int], str] | None = None,
    cell_extra: Mapping[tuple[int, int], str] | None = None,
    group_row: str | None = None,
    key_column: str | None = None,
) -> str:
    """A scroll container holding one table: the page never scrolls sideways, the table does."""
    help = help or {}
    cell_titles = cell_titles or {}
    cell_extra = cell_extra or {}
    key_index = block.columns.index(key_column) if key_column in block.columns else None
    head = "".join(
        f'<th scope="col"{_key_attr(index == key_index)}{_title_attr(help.get(column))}>'
        f"{escape(column)}</th>"
        for index, column in enumerate(block.columns)
    )
    rows = "".join(
        _row(index, row, cell_titles, cell_extra, key_index) for index, row in enumerate(block.rows)
    )
    thead = f"{group_row or ''}<tr>{head}</tr>"
    return f'<div class="scroll"><table><thead>{thead}</thead><tbody>{rows}</tbody></table></div>'


def _row(
    row_index: int,
    row: Sequence[str],
    cell_titles: Mapping[tuple[int, int], str],
    cell_extra: Mapping[tuple[int, int], str],
    key_index: int | None = None,
) -> str:
    cells: list[str] = []
    for column_index, cell in enumerate(row):
        key = _key_attr(column_index == key_index)
        title = _title_attr(cell_titles.get((row_index, column_index)))
        extra = cell_extra.get((row_index, column_index), "")
        cells.append(f"<td{key}{title}>{escape(cell)}{extra}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def _key_attr(is_key: bool) -> str:
    return ' class="key"' if is_key else ""


def _title_attr(text: str | None) -> str:
    return f' title="{escape(text, quote=True)}"' if text else ""


def theme_switch() -> str:
    """Auto / light / dark as three radio buttons the stylesheet reads with `:has()`.

    The page ships no script, so the choice lives in the radios themselves: `auto` (checked
    on open) leaves the palette to the system preference, the other two override it for
    this view. Nothing is remembered between opens, which is the price of staying static.
    """
    choices = (("auto", "Auto", " checked"), ("light", "Light", ""), ("dark", "Dark", ""))
    labels = "".join(
        f'<label><input type="radio" name="theme" id="theme-{value}"{checked}>'
        f"<span>{word}</span></label>"
        for value, word, checked in choices
    )
    return f'<fieldset class="theme"><legend>Theme</legend>{labels}</fieldset>'


def legend(labels: Sequence[str]) -> str:
    items = "".join(
        f'<li><span class="swatch s{index + 1}"></span>{escape(label)}</li>'
        for index, label in enumerate(labels)
    )
    return f'<ul class="legend">{items}</ul>'


def chart(svg: str, caption_html: str, unmeasured: Sequence[str] = ()) -> str:
    """The scroll container, why some rungs have no bar, then the caption.

    `caption_html` is markup, not user text -- every caller builds it from a fixed template
    (interpolating only numbers and pre-escaped labels), the same way `ENDPOINT_CAPTION_HTML`
    carries its own `<code>` tag.
    """
    aside = f'<p class="caption">Not charted</p>{notes(unmeasured)}' if unmeasured else ""
    return f'<div class="chart-scroll">{svg}</div>{aside}<figcaption>{caption_html}</figcaption>'


def notes(lines: Sequence[str]) -> str:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    return f'<ul class="notes">{items}</ul>'


def footer(document: Document) -> str:
    """The run directory path (item 13 moves it out of the header)."""
    where = escape(relative_run(document.get("run_dir")))
    return f'<footer class="foot"><p class="where">{where}</p></footer>'
