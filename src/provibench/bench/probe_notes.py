"""The notes a probe summary owes its reader, beyond the numbers themselves.

A record's own `note` (a fallback taken, a burst delivery), the run-wide skip note, and the
"more than one model" note are each computed from the raw records; `summarize_probe` folds
the result into `ProbeSummary.notes`. Kept apart from `probe_summary` purely for its line
budget -- these functions do not need anything `probe_summary` does not already export
elsewhere, so there is no import to route around.
"""

from __future__ import annotations

from collections.abc import Sequence

from provibench.bench.probe_errors import skip_note
from provibench.bench.probe_models import ProbeResult, is_failed
from provibench.bench.probe_stream import BURST_NOTE

__all__ = ["probe_notes"]

_ROLE_WORDS: dict[str, tuple[str, str]] = {
    "cold": ("cold write", "cold writes"),
    "warm": ("warm read", "warm reads"),
    "ttl": ("ttl read", "ttl reads"),
    "stream": ("throughput request", "throughput requests"),
    "precheck": ("pre-check request", "pre-check requests"),
}
_ROLE_ORDER = ("cold", "warm", "stream", "ttl", "precheck")
_OK_MIN, _OK_MAX = 200, 300
"""The 2xx band `probe_models.is_served` reads a status against."""


def probe_notes(records: Sequence[ProbeResult], models: Sequence[str]) -> list[str]:
    """What a spec's records amount to beyond the numbers: the notes a reader is owed.

    Every renderer already prefixes a note with its endpoint's own label (the text report's
    `label: note`, the HTML page's bold short label), so a note here reads as a sentence
    with that prefix as its subject, never repeating the endpoint's name inside itself.
    """
    notes = _record_notes(records)
    skip = skip_note(records)
    for note in (skip, _burst_note(records)):
        if note:
            notes.append(note)
    # A skipped cold write is already explained by `skip`; counting it again here would say
    # the same failure twice. A warm, ttl or stream failure never gets a `skip` note, so it
    # would otherwise go unexplained next to a bare `errors` count.
    reason_records = records if skip is None else [r for r in records if r.role != "cold"]
    notes.extend(_error_reason_notes(reason_records))
    fallbacks = sum(1 for r in records if r.cached == 0 and (r.native_tokens_cached or 0) > 0)
    if fallbacks:
        notes.append(_fallback_note(fallbacks))
    if len(models) > 1:
        notes.append(f"responses named more than one model: {', '.join(models)}")
    return notes


def _error_reason_notes(records: Sequence[ProbeResult]) -> list[str]:
    """One note per (role, reason) group of failures, so a bare `errors` count is never
    silent about what actually failed."""
    groups: dict[tuple[str, str], int] = {}
    for record in records:
        if is_failed(record):
            key = (record.role, _failure_reason(record))
            groups[key] = groups.get(key, 0) + 1
    ordered = sorted(
        groups,
        key=lambda key: (
            _ROLE_ORDER.index(key[0]) if key[0] in _ROLE_ORDER else len(_ROLE_ORDER),
            key[1],
        ),
    )
    return [_error_reason_line(role, reason, groups[role, reason]) for role, reason in ordered]


def _failure_reason(record: ProbeResult) -> str:
    """Why the request produced no usable answer, in the reader's terms.

    A gateway answers `200` and puts the upstream failure in the body often enough that the
    status alone would say `failed (HTTP 200)`, which reads as a success and a failure at
    once; a 2xx that still carried an error payload is named for what it was instead.
    """
    if _OK_MIN <= record.status < _OK_MAX:
        return "provider error"
    return f"HTTP {record.status}" if record.status > 0 else "connection error"


def _error_reason_line(role: str, reason: str, count: int) -> str:
    singular, plural = _ROLE_WORDS.get(role, (f"{role} request", f"{role} requests"))
    words = singular if count == 1 else plural
    return f"{count} {words} failed ({reason})"


def _fallback_note(fallbacks: int) -> str:
    plural = "request" if fallbacks == 1 else "requests"
    return (
        f"did not report the cached share for {fallbacks} {plural}; using OpenRouter's own "
        "billing record instead."
    )


def _record_notes(records: Sequence[ProbeResult]) -> list[str]:
    """One note per distinct record note, without the burst marker.

    A burst record's own note says `burst` and nothing else, and the rung it happened on is
    what a reader needs; `_burst_note` writes that in its place.
    """
    notes: list[str] = []
    for record in records:
        for part in _note_parts(record.note or ""):
            if part != BURST_NOTE and part not in notes:
                notes.append(part)
    return notes


def _note_parts(note: str) -> list[str]:
    """One record note split into its parts: `append_note` joins them with a semicolon."""
    return [part.strip() for part in note.split(";") if part.strip()]


def _burst_note(records: Sequence[ProbeResult]) -> str | None:
    """Why the throughput chart has a blank `tok/s` cell: one flush, not a stream."""
    rungs = sorted(
        {record.rung for record in records if BURST_NOTE in _note_parts(record.note or "")}
    )
    if not rungs:
        return None
    if len(rungs) == 1:
        return (
            "sent its whole answer in one burst; tok/s could not be measured, so the cell is blank."
        )
    names = ", ".join(str(rung) for rung in rungs)
    return (
        f"sent its whole answer in one burst on rungs {names}; tok/s could not be measured "
        "there, so the cell may be blank."
    )
