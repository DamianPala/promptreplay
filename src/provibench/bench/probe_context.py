"""The per-spec context a probe's requests share, and what building a record needs of it.

Its own module so the request path (`probe_requests`, `probe_stream`) can depend on the
context without importing the protocol that drives it: `bench/probe.py` stays the only
module that knows the order requests are sent in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from provibench.bench.probe_models import ProbeResult
from provibench.bench.replay import ReplayOptions
from provibench.bench.targets import RunSpec

__all__ = ["ProbeContext"]


@dataclass(frozen=True, slots=True)
class ProbeContext:
    """One spec's probe: where its requests go, how they are shaped, and what reports them.

    `opts` shapes the recorded body (the `max_tokens` a read sends, thinking stripping);
    `throughput` and `ttl_s` are the protocol's own switches, so the module that runs the
    protocol reads them off this object rather than carrying them through every call.
    """

    spec: RunSpec
    opts: ReplayOptions
    api_key: str
    client: httpx.AsyncClient
    throughput: bool = True
    ttl_s: list[int] | None = None
    progress: Callable[[ProbeResult], None] | None = None
