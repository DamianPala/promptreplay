"""The clock boundary: wall time, sleeping, and command deadlines (I8)."""

import time
from typing import Protocol


class Clock(Protocol):
    """Wall-clock time and sleeping, injected so tests control both."""

    def now(self) -> float:
        """Seconds since the Unix epoch."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for `seconds`."""
        ...


class SystemClock:
    """The real clock."""

    def now(self) -> float:
        """Seconds since the Unix epoch."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Block for `seconds`."""
        time.sleep(seconds)


class Deadline:
    """A command's deadline: starts at the first platform attempt, bounds everything after.

    `limit` is `None` for a wait that is unbounded by default (I8b).
    """

    def __init__(self, clock: Clock, limit: float | None) -> None:
        self._clock = clock
        self._limit = limit
        self._started_at: float | None = None

    def start(self) -> None:
        """Start the clock; later calls change nothing."""
        if self._started_at is None:
            self._started_at = self._clock.now()

    def remaining(self) -> float | None:
        """Seconds left, `None` when unbounded, never negative."""
        if self._limit is None:
            return None
        self.start()
        elapsed = self._clock.now() - (self._started_at or 0.0)
        return max(0.0, self._limit - elapsed)

    def expired(self) -> bool:
        """Whether a bounded deadline has passed."""
        remaining = self.remaining()
        return remaining is not None and remaining <= 0.0

    def budget(self, request_budget: float) -> float:
        """The timeout for one request: the request budget, cut by the deadline."""
        remaining = self.remaining()
        if remaining is None:
            return request_budget
        return min(request_budget, remaining)
