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
from provibench.bench.probe_models import ProbeResult
from provibench.bench.probe_stream import BURST_NOTE

__all__ = ["probe_notes"]


def probe_notes(records: Sequence[ProbeResult], models: Sequence[str]) -> list[str]:
    """What a spec's records amount to beyond the numbers: the notes a reader is owed."""
    notes = _record_notes(records)
    for note in (skip_note(records), _burst_note(records)):
        if note:
            notes.append(note)
    fallbacks = sum(1 for r in records if r.cached == 0 and (r.native_tokens_cached or 0) > 0)
    if fallbacks:
        notes.append(f"cached taken from the OpenRouter generation for {fallbacks} request(s)")
    if len(models) > 1:
        notes.append(f"responses named more than one model: {', '.join(models)}")
    return notes


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
    """One line naming every rung whose generation arrived in a single flush."""
    rungs = sorted(
        {record.rung for record in records if BURST_NOTE in _note_parts(record.note or "")}
    )
    if not rungs:
        return None
    which = "rung" if len(rungs) == 1 else "rungs"
    return f"burst delivery on {which} {', '.join(str(rung) for rung in rungs)}"
