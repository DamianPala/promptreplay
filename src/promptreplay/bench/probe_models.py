"""The probe protocol's vocabulary: what a run is told to do, and what it records.

`bench/probe.py` owns the protocol itself (rungs, the cold write, the warm reads, the
throughput request, the TTL re-reads) and `bench/probe_requests.py` performs its requests;
these are the shapes they pass around, kept apart so those modules stay readable files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from promptreplay.bench.replay import ReplayOptions, ReplayResult
from promptreplay.bench.rungs import broadcast_repeats, select_rungs, validate_rungs
from promptreplay.bench.trace import TraceEntry, Usage

type Role = Literal["cold", "warm", "stream", "ttl", "precheck"]
"""What a probe request is for: the write, a warm read, the throughput request, a TTL read,
or an availability pre-check -- never part of a spec's own records, persisted separately."""

_DEFAULT_REPEATS = [6, 2, 2]
_TWOXX_MIN = 200
_TWOXX_MAX = 300


@dataclass(frozen=True, slots=True)
class ProbeCall:
    """Which request this is, for the record it produces."""

    rung: int
    role: Role
    attempt: int
    nonce: str | None


@dataclass(slots=True)
class ProbeOutcome:
    """A record and the replay result that carries its OpenRouter generation lookup."""

    record: ProbeResult
    replay: ReplayResult | None = None
    thinking_dropped: bool = False
    """The request carried a `thinking` parameter that had to go for it to succeed."""


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
    throughput: bool = True
    """Send the streamed generation request after each rung's warm reads."""
    ttl_s: list[int] | None = None
    """Seconds after the first served rung's last warm read to re-read its cache; `None` = off."""

    @model_validator(mode="after")
    def _check(self) -> ProbeOptions:
        if self.ttl_s is not None:
            if not self.ttl_s:
                raise ValueError("--ttl must contain at least one offset in seconds")
            if any(offset <= 0 for offset in self.ttl_s):
                raise ValueError("--ttl offsets must be positive seconds")
            if self.ttl_s != sorted(set(self.ttl_s)):
                raise ValueError("--ttl offsets must ascend, each given once")
            too_close = next((offset for offset in self.ttl_s if offset <= self.gap_s), None)
            if too_close is not None:
                raise ValueError(
                    f"--gap {self.gap_s:g} is larger than the TTL offset {too_close}s, "
                    "so the first warm read would already sit past it"
                )
        return self

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
    """One probe request: a rung's cold write, one of its warm reads, or its extra requests.

    The streamed throughput request fills `ttft_ms`, `gen_tok_s` and `fingerprint` — its
    output token count rides in `usage` like every other request's — and the TTL re-reads
    put their requested offset in `attempt` and the offset they actually landed on in
    `offset_actual_s`. Everything else is filled the same way for every role.
    """

    spec_label: str
    rung: int
    role: Role
    attempt: int
    seq: int
    status: int
    latency_ms: float
    ttft_ms: float | None = None
    gen_tok_s: float | None = None
    fingerprint: str | None = None
    offset_actual_s: float | None = None
    """A TTL read's real offset: seconds from the warm baseline to when it was sent."""
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
