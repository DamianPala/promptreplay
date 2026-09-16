"""LiteLLM's community price table: fetch it, cache it, and resolve a native target's prices.

Native targets have no pricing API, so their prices come from LiteLLM's
`model_prices_and_context_window.json` — one JSON object keyed by `<provider>/<model>` and by
bare model name, each entry carrying USD-per-token costs. The whole file is cached under the
XDG cache directory, so a run reads the copy it has and touches the network only when that
copy is missing or older than a week.

Resolution is exact and two-step, never fuzzy: `<litellm_provider>/<model>` for a target that
declares a provider, then the bare `<model>`. A `targets.toml` `prices` entry is the override
and wins over both: that is the path for a model LiteLLM does not list, and for a rate that
is not the listed one — LiteLLM carries peak rates, and DeepSeek for one halves them
off-peak.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field

from provibench.bench.targets import Prices, Target
from provibench.core.settings import XdgPath

URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
"""The community table: every model LiteLLM prices, USD per token."""

CACHE = XdgPath("XDG_CACHE_HOME", ".cache", "provibench/litellm-prices.json")
"""Where the table is cached, resolved against the invocation's environment and home."""

MAX_AGE_S = 7 * 24 * 60 * 60
"""How long a cached copy counts as fresh; past it, the next caller refetches."""

SOURCE_TARGETS = "targets"
"""The price came from the target's own `targets.toml` `prices` entry."""

SOURCE_LITELLM = "litellm"
"""The price came from the community table."""

SCHEMA_EXAMPLE_KEY = "sample_spec"
"""The key the table carries as its schema example: documentation, not a model.

Its costs are a real `0.0` and its token limits are descriptive strings, so no structural
check catches it; LiteLLM's own loader drops the key by name, and so does this one, because
a lookup that resolved it would report a model priced at nothing.
"""

_FETCH_TIMEOUT_S = 30.0


def cache_path(env: Mapping[str, str], home: Path) -> Path:
    """The cache file: `$XDG_CACHE_HOME/provibench/litellm-prices.json` or the home fallback."""
    return CACHE.resolve(env, home)


class PriceTable(BaseModel):
    """LiteLLM's table, reduced to the keys it prices and the four costs this project uses."""

    prices: dict[str, Prices] = Field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PriceTable:
        """The entries of the raw table that can price a prompt, keyed as LiteLLM keys them.

        An entry without a numeric `input_cost_per_token` prices nothing and is dropped:
        the table carries provider metadata objects whose keys a lookup must never fall back
        onto, and the schema example goes with them by name. A missing or null cache cost is
        a real 0.0 — most providers do not bill cache writes separately — and the irrelevant
        fields (`mode`, context limits, capability flags) are not read at all.
        """
        prices: dict[str, Prices] = {}
        for key, entry in payload.items():
            if key == SCHEMA_EXAMPLE_KEY or not isinstance(entry, dict):
                continue
            costs = cast("dict[str, Any]", entry)
            input_price = _cost(costs.get("input_cost_per_token"))
            if input_price is None:
                continue
            prices[key] = Prices(
                input=input_price,
                cache_read=_cost(costs.get("cache_read_input_token_cost")) or 0.0,
                cache_write=_cost(costs.get("cache_creation_input_token_cost")) or 0.0,
                output=_cost(costs.get("output_cost_per_token")) or 0.0,
            )
        return cls(prices=prices)

    def resolve(self, kind_or_provider: str | None, model: str) -> Prices | None:
        """The prices of `<kind_or_provider>/<model>`, else of the bare `<model>`, else `None`.

        `kind_or_provider` is what LiteLLM namespaces the model under: usually the target's
        `litellm_provider` (`deepseek`), and its kind when a target declares no provider.
        `None` tries the bare model name alone. Both lookups are exact: a near miss is a
        wrong price, and a wrong price is worse than none.
        """
        keys = [f"{kind_or_provider}/{model}", model] if kind_or_provider else [model]
        for key in keys:
            found = self.prices.get(key)
            if found is not None:
                return found
        return None


def resolve_target(
    target: Target, model: str, table: PriceTable | None
) -> tuple[Prices, str] | None:
    """A native target's prices and their source, or `None` when nothing prices them.

    The target's own `prices` entry is the override and wins: it is the one place a user can
    state the rate they are really billed, which for an off-peak discount is half of what
    the community table lists. Only then is the LiteLLM table consulted.
    """
    own = target.prices.get(model)
    if own is not None:
        return own, SOURCE_TARGETS
    if table is None:
        return None
    found = table.resolve(target.litellm_provider, model)
    return None if found is None else (found, SOURCE_LITELLM)


def load_table(path: Path) -> PriceTable:
    """The table stored at `path`; `ValueError` when it is missing or not that JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: the price table is not a JSON object")
    return PriceTable.from_payload(cast("dict[str, Any]", payload))


def cache_age_s(path: Path, now: float) -> float | None:
    """How long ago the cached copy was written, or `None` when there is no copy.

    The file's mtime is the fetch time, which is the whole of the bookkeeping: the cache is
    the table LiteLLM served, unchanged, so nothing else has to be stored beside it.
    """
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


async def fetch_payload(client: httpx.AsyncClient) -> dict[str, Any]:
    """The table as the network serves it; `httpx.HTTPError`/`ValueError` when it does not."""
    resp = await client.get(URL)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{URL} did not return a JSON object")
    return cast("dict[str, Any]", payload)


async def _fetch() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
        return await fetch_payload(client)


def write_table(path: Path, payload: Mapping[str, Any]) -> None:
    """Store the payload as the cache file, creating its directory; its mtime is the fetch time.

    Written next to the target and renamed over it, so an interrupted write or two runs
    fetching at once never leave a truncated copy where the offline fallback expects one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    partial.replace(path)


def refresh(path: Path) -> PriceTable:
    """Fetch the table, rewrite the cache file and return what was fetched."""
    payload = asyncio.run(_fetch())
    write_table(path, payload)
    return PriceTable.from_payload(payload)


@dataclass(frozen=True, slots=True)
class Cached:
    """One usable table: where it came from, how old it is, and what the call did."""

    table: PriceTable
    path: Path
    age_s: float
    fetched: bool
    warning: str | None = None


class TableUnavailable(Exception):
    """The table could not be fetched and there was no cached copy to fall back on.

    `line` is what a caller that can carry on says in one line; the message and the hint are
    what a caller that cannot turns into an F3 failure. Both name the URL and the
    `targets.toml` entry that prices a model without the table at all.
    """

    def __init__(self, reason: str, *, path: Path) -> None:
        super().__init__(f"The LiteLLM price table at {URL} could not be read: {reason}")
        self.reason = reason
        self.path = path

    @property
    def hint(self) -> str:
        """Where to go from here: the network, or the override that needs neither."""
        return (
            "Retry when the network is up, or price the model from targets.toml instead: "
            '[targets.<name>.prices."<model>"] with input, cache_read, cache_write and '
            f"output in USD per 1M tokens (cache file: {self.path})"
        )

    @property
    def line(self) -> str:
        """One line for a run that keeps going with the model priced as `n/a`."""
        return (
            f"no LiteLLM price table at {URL} ({self.reason}); native prices stay n/a — a "
            "targets.toml prices entry states one by hand"
        )


def cached_table(path: Path, *, update: bool) -> Cached:
    """The table to use: a fresh cache, a refetch, or a stale copy and a warning.

    The cache is the network's only stand-in, so a failed refresh is never fatal while a
    copy is on disk: the copy is used and the caller is told how old it is. With no usable
    copy there is nothing to price with, and saying so beats a guessed rate.

    The file's mtime is compared against wall time rather than against an injected clock,
    because the thing being aged is a filesystem fact, like a run directory's own timestamp.
    """
    age = cache_age_s(path, time.time())
    if not update and age is not None and age <= MAX_AGE_S:
        cached = _read(path)
        if cached is not None:
            return Cached(table=cached, path=path, age_s=age, fetched=False)
    try:
        return Cached(table=refresh(path), path=path, age_s=0.0, fetched=True)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return _fallback(path, age, exc)


def _fallback(path: Path, age: float | None, exc: Exception) -> Cached:
    """The stale copy after a failed refresh, or `TableUnavailable` when there is none."""
    if age is not None:
        cached = _read(path)
        if cached is not None:
            return Cached(
                table=cached,
                path=path,
                age_s=age,
                fetched=False,
                warning=(
                    f"the LiteLLM price table at {URL} could not be refreshed ({exc}); "
                    f"using the copy fetched {format_age(age)}"
                ),
            )
    reason = str(exc)
    if age is not None:
        reason = f"{reason}; the cached copy at {path} is not a usable table"
    raise TableUnavailable(reason, path=path)


def _read(path: Path) -> PriceTable | None:
    """The cached table, or `None` when there is no usable copy at `path`."""
    try:
        return load_table(path)
    except ValueError:
        return None


def _cost(value: object) -> float | None:
    """One USD-per-token cost as USD per 1M tokens, or `None` when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) * 1e6


def format_age(seconds: float) -> str:
    """A one-phrase age for a message: `less than a minute ago`, `5 hours ago`, `3 days ago`."""
    if seconds < 60:
        return "less than a minute ago"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    days = hours // 24
    return f"{days} day{'s' if days > 1 else ''} ago"
