"""The report's charts: inline SVG built in Python, no library and no external assets.

Three forms, one per question the report answers: grouped bars for the cache hit rate of
every endpoint at every rung, horizontal bars for the effective prompt price of each
endpoint, and a line for one endpoint's cache curve across a full replay's turns.

Everything a mark needs to be readable on its own is in the markup: a `class` naming the
series (`s1`..`s8`), which the page's stylesheet maps to a light or dark value, and a
`<title>` carrying the exact numbers. There is no `xmlns` on the root element — an inline
SVG in an HTML document is already in the SVG namespace, and the report promises to
reference nothing over the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

MAX_SERIES = 8
"""How many endpoints the grouped chart colours: the palette's validated slots, in order."""

_CHART_WIDTH = 1080
"""The user units one chart is laid out in; CSS scales it to the page."""

_PAD_LEFT = 56
_PAD_RIGHT = 20
_PAD_TOP = 16
_PAD_BOTTOM = 44
_GROUP_FILL = 0.84
"""Share of a group's column the bars occupy; the rest is the gap between groups."""
_BAR_FILL = 0.86
"""Share of its slot one bar fills; the rest is the 2 px surface gap between neighbours."""
_MAX_BAR = 120
"""Widest a bar grows: with one or two groups an uncapped slot would make blocks, not bars."""
_BAR_RADIUS = 4
"""Rounded data-end radius, clamped so a short bar keeps its shape."""
_ZERO_HEIGHT = 2
"""A measured zero still gets a mark on the baseline, so a series never silently vanishes."""
_AXIS_TITLE_BAND = 14
"""Room above the plot for a y-axis title, so it does not touch the topmost tick label."""
_COUNT_STEPS = (0, 25, 50, 75, 100)
"""The percentage gridlines both percentage charts share."""

_ROW_HEIGHT = 26
_ROW_BAR = 11
_LABEL_COLUMN = 236
_VALUE_COLUMN = 64
_LABEL_CHARS = 34
"""An endpoint label is clipped to this many characters before the bars start."""
_MARKER_RADIUS = 4
_MAX_X_LABELS = 10
_ELLIPSIS = "…"


@dataclass(frozen=True, slots=True)
class Bar:
    """One bar: which series it belongs to, its 0..1 value, and what hovering it says."""

    series: int
    value: float
    title: str


@dataclass(frozen=True, slots=True)
class BarGroup:
    """One x position: its tick label and one slot per series, `None` where it has no bar."""

    label: str
    bars: Sequence[Bar | None]


@dataclass(frozen=True, slots=True)
class BarRow:
    """One horizontal bar: its series, row label, value (`None` draws `-`), value text
    and hover text."""

    series: int
    label: str
    value: float | None
    value_text: str
    title: str


@dataclass(frozen=True, slots=True)
class Point:
    """One turn of a cache curve: its 1-based index, its 0..1 value, and its hover text."""

    index: int
    value: float
    title: str


def grouped_bars(
    groups: Sequence[BarGroup], *, label: str, y_axis: str, x_axis: str, height: int = 340
) -> str:
    """Vertically scaled bars, grouped by x position: one slot per series in every group.

    A series keeps its slot across every group, so it can be followed by eye even where
    another series has no bar; a bar is capped in width so that two series do not become
    two blocks. A measured zero keeps a mark on the baseline, and a `None` slot is simply
    left empty rather than drawn as one.
    """
    if not groups:
        return ""
    slots = max(len(group.bars) for group in groups)
    top = _PAD_TOP + _AXIS_TITLE_BAND
    plot_width = _CHART_WIDTH - _PAD_LEFT - _PAD_RIGHT
    plot_height = height - top - _PAD_BOTTOM
    group_width = plot_width / len(groups)
    slot = min(group_width * _GROUP_FILL / slots, _MAX_BAR / _BAR_FILL)
    inner = slot * slots
    bar_width = slot * _BAR_FILL
    baseline = top + plot_height

    parts = [_grid(plot_height, top), _baseline(baseline)]
    parts.append(_text(_PAD_LEFT - 10, top - 8, y_axis, "axis-title"))
    for index, group in enumerate(groups):
        left = _PAD_LEFT + index * group_width + (group_width - inner) / 2
        parts.append(_tick(_PAD_LEFT + (index + 0.5) * group_width, baseline + 20, group.label))
        for position, bar in enumerate(group.bars):
            if bar is None:
                continue
            bar_height = max(_ZERO_HEIGHT, plot_height * _clamp(bar.value))
            x = left + position * slot + (slot - bar_width) / 2
            parts.append(
                _path(
                    _rounded(x, baseline - bar_height, bar_width, bar_height, "top"),
                    f"bar s{bar.series + 1}",
                    bar.title,
                )
            )
    parts.append(_text(_CHART_WIDTH - _PAD_RIGHT, baseline + 38, x_axis, "axis-title"))
    return _svg(parts, f"{label} 0 to 100 percent", height, label)


def horizontal_bars(rows: Sequence[BarRow], *, label: str) -> str:
    """One bar per row, every bar scaled to the largest value in the chart.

    A row without a value draws no bar and prints `-` where the value would be: an empty
    slot would read as a zero the run never measured.
    """
    if not rows:
        return ""
    height = _PAD_TOP + len(rows) * _ROW_HEIGHT + _PAD_BOTTOM - 24
    largest = max((row.value for row in rows if row.value is not None), default=0.0)
    span = _CHART_WIDTH - _LABEL_COLUMN - _PAD_RIGHT - _VALUE_COLUMN
    parts: list[str] = []
    for index, row in enumerate(rows):
        top = _PAD_TOP + index * _ROW_HEIGHT
        y = top + (_ROW_HEIGHT - _ROW_BAR) / 2
        parts.append(_row_label(row.label, y + _ROW_BAR - 2))
        if row.value is None:
            parts.append(_text(_LABEL_COLUMN, y + _ROW_BAR - 2, "-", "value", row.title))
            continue
        width = max(_ZERO_HEIGHT, span * (row.value / largest if largest > 0 else 0.0))
        css = f"bar s{row.series + 1}"
        parts.append(_path(_rounded(_LABEL_COLUMN, y, width, _ROW_BAR, "right"), css, row.title))
        parts.append(_text(_LABEL_COLUMN + width + 8, y + _ROW_BAR - 2, row.value_text, "value"))
    return _svg(parts, label, height, label)


def line_chart(points: Sequence[Point], *, label: str, height: int = 220) -> str:
    """One series over time: the turn index on x, the cached fraction on y.

    Each point is its own hover target, so a reader can read a single turn out of the
    curve; the line between them carries no marks of its own.
    """
    if not points:
        return ""
    plot_width = _CHART_WIDTH - _PAD_LEFT - _PAD_RIGHT
    plot_height = height - _PAD_TOP - _PAD_BOTTOM
    baseline = _PAD_TOP + plot_height
    span = max(1, len(points) - 1)
    xs = [_PAD_LEFT + index / span * plot_width for index in range(len(points))]
    ys = [baseline - plot_height * _clamp(point.value) for point in points]
    parts = [_grid(plot_height), _baseline(baseline)]
    parts.append(_path(_polyline(xs, ys), "line", None))
    for x, y, point in zip(xs, ys, points, strict=True):
        parts.append(_circle(x, y, _MARKER_RADIUS, "marker", point.title))
    for index in _x_indices(len(points)):
        parts.append(_tick(xs[index], baseline + 20, str(points[index].index)))
    parts.append(_text(_CHART_WIDTH - _PAD_RIGHT, baseline + 38, "turn", "axis-title"))
    return _svg(parts, f"{label} 0 to 100 percent cached", height, label)


def _x_indices(count: int) -> list[int]:
    """The points that get an x tick: evenly spaced, never more than `_MAX_X_LABELS`."""
    step = max(1, -(-count // _MAX_X_LABELS))
    ticks = list(range(0, count, step))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    return ticks


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _grid(plot_height: int, top: int = _PAD_TOP) -> str:
    """The percentage gridlines and their labels, behind every mark."""
    parts: list[str] = []
    for step in _COUNT_STEPS:
        y = top + plot_height * (1 - step / 100)
        parts.append(_text(_PAD_LEFT - 10, y + 4, str(step), "tick end"))
        parts.append(_line(_PAD_LEFT, y, _CHART_WIDTH - _PAD_RIGHT, "grid"))
    return "".join(parts)


def _baseline(y: float) -> str:
    return _line(_PAD_LEFT, y, _CHART_WIDTH - _PAD_RIGHT, "axis")


def _polyline(xs: Sequence[float], ys: Sequence[float]) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))


def _rounded(x: float, y: float, width: float, height: float, corners: str) -> str:
    """A bar path with two rounded corners: its data end, the end away from the baseline.

    A rectangle's `rx` would round all four, which reads as a floating capsule rather than
    a bar growing out of an axis, so the two corners at the baseline stay square.
    """
    radius = max(0.0, min(_BAR_RADIUS, width / 2, height / 2))
    if corners == "top":
        return (
            f"M{x:.1f},{y + height:.1f} V{y + radius:.1f} "
            f"Q{x:.1f},{y:.1f} {x + radius:.1f},{y:.1f} "
            f"H{x + width - radius:.1f} "
            f"Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + radius:.1f} "
            f"V{y + height:.1f} Z"
        )
    return (
        f"M{x:.1f},{y:.1f} H{x + width - radius:.1f} "
        f"Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + radius:.1f} "
        f"V{y + height - radius:.1f} "
        f"Q{x + width:.1f},{y + height:.1f} {x + width - radius:.1f},{y + height:.1f} "
        f"H{x:.1f} Z"
    )


def _row_label(label: str, y: float) -> str:
    """A row's label, carrying the full text on hover when the column clipped it.

    The clipped text says a name was cut and the hover says what it was, so a long endpoint
    id is never lost to the column width — the bar beside it can only be identified by it.
    """
    shown = _clip(label)
    title = _title(label) if shown != label else ""
    return f'<text class="row-label" x="0.0" y="{y:.1f}">{title}{escape(shown)}</text>'


def _clip(label: str) -> str:
    """A row label cut to `_LABEL_CHARS`, with an ellipsis when it did not fit."""
    if len(label) <= _LABEL_CHARS:
        return label
    return f"{label[: _LABEL_CHARS - len(_ELLIPSIS)]}{_ELLIPSIS}"


def _svg(parts: Sequence[str], aria: str, height: int, label: str) -> str:
    """The chart element: a responsive viewBox, a descriptive label, and the marks."""
    return (
        f'<svg class="chart" viewBox="0 0 {_CHART_WIDTH} {height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="{escape(aria, quote=True)}">'
        f"<title>{escape(label)}</title>{''.join(parts)}</svg>"
    )


def _text(x: float, y: float, text: str, css: str, title: str | None = None) -> str:
    return f'<text class="{css}" x="{x:.1f}" y="{y:.1f}">{_title(title)}{escape(text)}</text>'


def _line(x1: float, y1: float, x2: float, css: str) -> str:
    return f'<line class="{css}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y1:.1f}" />'


def _path(shape: str, css: str, title: str | None) -> str:
    return f'<path class="{css}" d="{shape}">{_title(title)}</path>'


def _circle(x: float, y: float, radius: float, css: str, title: str | None) -> str:
    return f'<circle class="{css}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}">{_title(title)}</circle>'


def _tick(x: float, y: float, text: str) -> str:
    return _text(x, y, text, "tick mid")


def _title(text: str | None) -> str:
    """The hover text of one mark; empty when the caller passed none."""
    if text is None:
        return ""
    return f"<title>{escape(text)}</title>"
