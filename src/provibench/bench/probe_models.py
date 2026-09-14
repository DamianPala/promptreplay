"""The probe protocol's vocabulary: what a run is told to do, and what it records.

`bench/probe.py` owns the protocol itself (rungs, the cold write, the warm reads, the
retries) and the engine that performs it; these are the shapes it works with, kept apart so
that module stays a single readable file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from provibench.bench.replay import ReplayOptions
from provibench.bench.rungs import broadcast_repeats, select_rungs, validate_rungs
from provibench.bench.trace import TraceEntry, Usage

type Role = Literal["cold", "warm"]

_DEFAULT_REPEATS = [6, 2, 2]
_TWOXX_MIN = 200
_TWOXX_MAX = 300


class ProbeOptions(BaseModel):
    """What one probe run does; `rungs` and `repeats` are resolved before any request."""

    rungs: list[int] | None = None
    """1-based turn indices; `None` picks the smallest, middle and largest turn."""
    repeats: list[int] = Field(default_factory=lambda: list(_DEFAULT_REPEATS))
    gap_s: float = 1.0
    warm: bool = False
    """Skip the nonce and read the cache as found, contamination included."""
    timeout_s: float = 300.0
    strip_thinking: bool = False

    def replay_options(self) -> ReplayOptions:
        """The options that shape the bodies: one output token, non-streaming."""
        return ReplayOptions(
            max_tokens=1, strip_thinking=self.strip_thinking, timeout_s=self.timeout_s
        )

    def resolved(self, conversation: Sequence[TraceEntry]) -> ProbeOptions:
        """This object with concrete rungs and one repeat count per rung, or raise.

        Repeated rungs collapse to one: the second cold request of a rung would read the
        first one's cache and could not be reported as cold.
        """
        rungs = select_rungs(conversation) if self.rungs is None else sorted(set(self.rungs))
        validate_rungs(rungs, conversation)
        return self.model_copy(
            update={"rungs": rungs, "repeats": broadcast_repeats(rungs, self.repeats)}
        )


class ProbeResult(BaseModel):
    """One probe request: a rung's cold write, or one of its warm reads."""

    spec_label: str
    rung: int
    role: Role
    attempt: int
    seq: int
    status: int
    latency_ms: float
    message_id: str | None = None
    model: str | None = None
    provider: str | None = None
    requested_providers: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    prompt_total: int = 0
    cached: int = 0
    cache_write: int = 0
    native_tokens_cached: int | None = None
    retries: int = 0
    error: str | None = None
    note: str | None = None
    generation: dict[str, Any] | None = None


class ProbeRun(BaseModel):
    """Everything one probe run produced, before it is persisted or summarised."""

    run_hex: str
    options: ProbeOptions
    records: dict[str, list[ProbeResult]] = Field(default_factory=dict)


def is_served(record: ProbeResult) -> bool:
    """Whether the provider answered this request with a 2xx and no error payload."""
    return record.error is None and _TWOXX_MIN <= record.status < _TWOXX_MAX


def is_failed(record: ProbeResult) -> bool:
    """Whether this request produced no usable answer."""
    return not is_served(record)
