"""The human view of a sweep's selection: the listing before a run, the sentence after one.

`bench.selection` decides which endpoints survive and owns the record of that (`SweepInfo`);
this module turns the same decision into words — `render_candidates` for the command that is
about to spend money, `selection_line` and `not_probed_lines` for the report of a run that
already did. Split out of `bench.selection` purely for that module's line budget.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from provibench.bench.labels import join_and, text_table
from provibench.bench.openrouter import Endpoint
from provibench.bench.selection import (
    UPTIME_FLOOR,
    Selection,
    SelectionDrop,
    SweepInfo,
    prefixed,
)

__all__ = [
    "not_probed_lines",
    "render_candidates",
    "selection_line",
]

_FLOOR_MARGIN = 0.1
"""How close to the floor an uptime has to be before the table prints it to two decimals."""

_CANDIDATE_COLUMNS = ("tag", "quant", "status", "uptime 1d", "$/M in", "lat p50 ms", "tput p50")


def render_candidates(
    selection: Selection,
    *,
    model: str,
    sort: str,
    zdr: bool,
    uptime_floor: float = UPTIME_FLOOR,
    quantization: Sequence[str] = (),
    pin: Sequence[str] = (),
) -> list[str]:
    """The listing printed before the estimate: the candidates, then what the criteria cut."""
    kept = len(selection.kept)
    header = (
        f"candidates: {model}, {kept} endpoint(s), sort={sort}, zdr={_on_off(zdr)}, "
        f"min-uptime={uptime_floor:g}"
    )
    if quantization:
        header += f", quant={','.join(quantization)}"
    if pin:
        header += f", pin={','.join(pin)}"
    rows = [_candidate_row(endpoint, uptime_floor) for endpoint in selection.kept]
    lines = [header, *text_table(_CANDIDATE_COLUMNS, rows)]
    if selection.dropped:
        lines.append("dropped: " + ", ".join(_drop_cell(drop) for drop in selection.dropped))
    return lines


def selection_line(sweep: SweepInfo) -> str:
    """A sentence: how many endpoints the run took and by what, then why the listing itself
    dropped any candidate (the stability floor, the ZDR filter or the quantization filter,
    in the reader's words).

    When `--include` named the endpoints outright, the sentence says so instead of claiming
    a ranking that never happened: "the 7 cheapest by listed price" would be false when the
    caller picked the tags themselves. A `--pin` addition is named the same way, on top of
    whichever clause applies.

    A pre-check's own drops are `not_probed_lines`'s job, kept separate because a pre-check
    spends money to find them and a listing-time drop costs nothing.
    """
    if sweep.included:
        pinned = set(sweep.pinned)
        matched = sum(
            1
            for entry in sweep.ranking
            if prefixed(entry.tag, sweep.included) and entry.tag not in pinned
        )
        noun = "endpoint" if matched == 1 else "endpoints"
        which = f"the {matched} {noun} named with --include"
    elif sweep.top is None:
        which = f"every candidate by {sweep.sort}"
    elif sweep.sort == "price":
        which = f"the {sweep.top} cheapest by listed price"
    else:
        which = f"the {sweep.top} best by {sweep.sort}"
    line = f"Of the OpenRouter providers, the run took {which}"
    if sweep.pinned:
        line += f", plus {len(sweep.pinned)} pinned"
    if sweep.zdr:
        line += ", zero-data-retention endpoints only"
    line += "."
    sentences = _drop_sentences(
        (drop for drop in sweep.dropped if not drop.checked), sweep.uptime_floor, sweep.quantization
    )
    if sentences:
        line += " " + " ".join(sentences)
    return line


def not_probed_lines(sweep: SweepInfo) -> list[str]:
    """One sentence per reason the availability pre-check skipped a candidate.

    Grouped the same way `selection_line` groups its own drops, but a pre-check failure
    keeps its own gateway reason rather than one of the listing's short codes.
    """
    return _drop_sentences(
        (drop for drop in sweep.dropped if drop.checked), sweep.uptime_floor, sweep.quantization
    )


def _drop_sentences(
    dropped: Iterable[SelectionDrop], floor: float, quantization: Sequence[str] = ()
) -> list[str]:
    """`{label} was skipped because {reason}.`, endpoints sharing a reason joined into one
    sentence, named the way the table above it does (`_short_drop_label`)."""
    groups: dict[str, list[str]] = {}
    for drop in dropped:
        groups.setdefault(_reason_words(drop, floor, quantization), []).append(
            _short_drop_label(drop)
        )
    sentences: list[str] = []
    for reason, labels in groups.items():
        verb = "was skipped" if len(labels) == 1 else "were skipped"
        sentences.append(f"{join_and(labels)} {verb} because {reason}.")
    return sentences


def _short_drop_label(drop: SelectionDrop) -> str:
    """`@tag` when the label has a provider suffix, else the label whole, as the table reads."""
    return f"@{drop.tag}" if "@" in drop.endpoint else drop.endpoint


def _reason_words(drop: SelectionDrop, floor: float, quantization: Sequence[str] = ()) -> str:
    """A drop's own code (`status -2`, `uptime 1d 77.90 %`, `quantization fp4`) in the
    reader's words.

    A pre-check failure already carries its provider's own gateway reason -- that is the
    "words" a reader can act on, and rewriting it would replace one true statement with a
    guess. A status code is left out: the `candidates:` listing and the run file keep it.

    The floor is named alongside the measured value: the number on its own reads as a fact
    about the endpoint, but the reason is that it fell short of a choice the tool made, not
    an OpenRouter rule, and a reader who set `--min-uptime` to something else needs to see
    which floor this run actually used. A quantization drop names the wanted values the same
    way, since `--quantization` is this run's own choice too.
    """
    if drop.checked:
        return drop.reason
    if drop.reason.startswith("status "):
        return "OpenRouter reported it as degraded"
    if drop.reason.startswith("uptime 1d "):
        pct = drop.reason.removeprefix("uptime 1d ").strip()
        return f"its one-day uptime ({pct}) was below the {floor:g} % floor"
    if drop.reason == "not ZDR":
        return "it is not a zero-data-retention endpoint"
    if drop.reason.startswith("quantization "):
        value = drop.reason.removeprefix("quantization ").strip()
        return f"its quantization ({value}) was not among {', '.join(quantization)}"
    return drop.reason


def _drop_cell(drop: SelectionDrop) -> str:
    """One dropped endpoint in the listing's own notation: `tag (reason)`."""
    return f"{drop.tag} ({drop.reason})"


def _candidate_row(endpoint: Endpoint, floor: float) -> list[str]:
    return [
        endpoint.tag,
        endpoint.quantization or "-",
        "-" if endpoint.status is None else str(endpoint.status),
        _uptime_cell(endpoint.uptime_1d, floor),
        f"{endpoint.prices.input:.3f}",
        _number(endpoint.latency_ms_30m, digits=0),
        _number(endpoint.throughput_30m, digits=1),
    ]


def _uptime_cell(value: float | None, floor: float) -> str:
    """The uptime column: one decimal, or two when the value sits next to the floor.

    `97.0` in this column could be anything from 96.95 up, and an endpoint kept by
    `--include` would then be printed as a value the floor should have dropped. The second
    digit is only paid where it disambiguates; a healthy endpoint reads as it always did.
    """
    if value is None:
        return "-"
    return f"{value:.{_uptime_digits(value, floor)}f}"


def _uptime_digits(value: float, floor: float) -> int:
    """How many decimals an uptime needs: two within `_FLOOR_MARGIN` of the floor, else one."""
    return 2 if abs(value - floor) < _FLOOR_MARGIN else 1


def _number(value: float | None, *, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _on_off(flag: bool) -> str:
    return "on" if flag else "off"
