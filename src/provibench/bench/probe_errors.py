"""What a provider's error payload says about a refusal, and how a summary states it.

An endpoint that is excluded by the account's OpenRouter settings, or a key out of quota,
fails every cold write of a spec before anything is cached: the row shows a count of errors
and no reason, which is not enough to act on. The error a record carries is the provider's
own JSON when the response had a body and a transport message when it did not; only the
former names a kind of refusal, and for a guardrail refusal it also names the setting to
change. This module turns that payload into the one line a summary carries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from provibench.bench.probe_models import ProbeResult

_REASONS = "reasons"
"""The word OpenRouter's guardrail refusal puts before the reasons it removed endpoints."""
_NOT_FOUND = "not_found"
"""The `error_type` an endpoint the account's settings exclude answers with."""
_MAX_REASON = 80
"""Longest guardrail reason a note repeats: the API's own sentence is longer than a line."""

__all__ = ["error_type", "guardrail_reason", "skip_note"]


def skip_note(records: Sequence[ProbeResult]) -> str | None:
    """`skipped, <error_type>` when every cold write failed the same way, else `None`.

    One line for the spec, not one per record: three rungs refused by the same setting are
    one thing to fix, and the summary already counts the errors that prove it failed.
    """
    colds = [record for record in records if record.role == "cold"]
    if not colds:
        return None
    kinds = {error_type(record.error) for record in colds}
    if len(kinds) != 1:
        return None
    [kind] = kinds
    if kind is None:
        return None
    reason = guardrail_reason(message_of(colds[0].error)) if kind == _NOT_FOUND else None
    return f"skipped, {kind}: {reason}" if reason else f"skipped, {kind}"


def message_of(error: str | None) -> str:
    """The sentence an error payload carries, or the error text when it carries none.

    A record's error is JSON for the whole payload, so its `message` field is where the
    provider's own words are — with the newlines the JSON escaped still escaped.
    """
    value = _payload(error).get("message")
    return value if isinstance(value, str) else (error or "")


def error_type(error: str | None) -> str | None:
    """The kind of refusal an error payload names, or `None` when it names none.

    A record's error is the provider's own JSON when the response carried one and a
    transport message when it did not; only the former says what went wrong, and the two
    spellings in use are OpenRouter's `error_type` and Anthropic's `type`.
    """
    payload = _payload(error)
    for key in ("error_type", "type"):
        value = payload.get(key)
        if isinstance(value, str) and value and value != "error":
            return value
    return None


def guardrail_reason(message: str) -> str | None:
    """The first reason OpenRouter's guardrail refusal names, capped at `_MAX_REASON`.

    That message ends `... for the following reasons (<tally>):\\n<reason>: <n> endpoint(s)
    excluded; configurable at <url>`, one line per reason. What a reader needs is the head
    of the first line — `Paid model training violation (account settings)` — because that
    is what the setting to change is named after.
    """
    _, marker, rest = message.partition(_REASONS)
    if not marker:
        return None
    tail = rest.lstrip()
    if tail.startswith("("):
        _, close, tail = tail.partition(")")
        if not close:
            return None
    first_line = next(
        (line.strip() for line in tail.lstrip(" :\n").splitlines() if line.strip()), ""
    )
    reason = first_line.partition(":")[0].strip()
    return reason[:_MAX_REASON] or None


def _payload(error: str | None) -> dict[str, Any]:
    """An error string as the JSON object it is, or an empty mapping when it is prose."""
    if not error:
        return {}
    try:
        value: object = json.loads(error)
    except json.JSONDecodeError:
        return {}
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}
