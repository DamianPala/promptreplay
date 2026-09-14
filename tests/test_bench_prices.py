"""Tests for provibench.bench.prices: the LiteLLM table, its cache and the resolver.

Every test here runs against the trimmed real fixture (`tests/fixtures/litellm-prices.json`)
or a small synthetic payload, and the fetch boundary is the one the session-wide
`no_price_network` fixture already made offline; a test that needs a successful fetch
replaces `provibench.bench.prices.fetch_payload` itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from provibench.bench.enrich import enrich_anthropic
from provibench.bench.prices import (
    MAX_AGE_S,
    PriceTable,
    TableUnavailable,
    cache_age_s,
    cached_table,
    format_age,
    load_table,
    resolve_target,
)
from provibench.bench.replay import ReplayResult
from provibench.bench.targets import Prices, RunSpec, Target
from provibench.bench.trace import Usage
from tests.conftest import FIXTURE_TABLE

_PAYLOAD: dict[str, Any] = {
    "deepseek/deepseek-flash": {
        "input_cost_per_token": 3e-7,
        "output_cost_per_token": 1.2e-6,
        "cache_read_input_token_cost": 6e-9,
        "cache_creation_input_token_cost": 0.0,
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "deepseek-chat": {
        "input_cost_per_token": 2.8e-7,
        "output_cost_per_token": 4.2e-7,
        "cache_read_input_token_cost": None,
    },
    "sample_spec": {"input_cost_per_token": 0.0, "mode": "sample_spec"},
    "no-prices-at-all": {"mode": "chat", "max_tokens": 8192},
}

_OWN = Prices(input=0.15, cache_read=0.003, cache_write=0.15, output=0.6)


def _target(
    name: str = "deepseek",
    *,
    model: str = "deepseek-flash",
    provider: str | None = "deepseek",
    prices: Mapping[str, Prices] | None = None,
) -> Target:
    return Target(
        name=name,
        url="https://api.test/v1/messages",
        api_key_env="KEY",
        kind="anthropic",
        prices={model: _OWN} if prices is None else dict(prices),
        litellm_provider=provider,
    )


def test_from_payload_reads_costs_as_usd_per_million() -> None:
    table = PriceTable.from_payload(_PAYLOAD)
    flash = table.prices["deepseek/deepseek-flash"]
    assert flash.input == pytest.approx(0.30)
    assert flash.output == pytest.approx(1.20)
    assert flash.cache_read == pytest.approx(0.006)
    assert flash.cache_write == 0.0


def test_a_null_cache_cost_becomes_zero_and_an_absent_one_too() -> None:
    table = PriceTable.from_payload(_PAYLOAD)
    # null in the payload, and missing entirely from the deepseek-flash-shaped one below
    assert table.prices["deepseek-chat"].cache_read == 0.0
    assert table.prices["deepseek-chat"].cache_write == 0.0
    assert PriceTable.from_payload({"m": {"input_cost_per_token": 1e-6}}).prices["m"].output == 0.0


def test_the_schema_example_and_unpriced_blobs_never_price_anything() -> None:
    """`sample_spec` costs a real 0.0 and lists placeholder strings, so it goes by name."""
    table = PriceTable.from_payload(_PAYLOAD)
    assert "sample_spec" not in table.prices
    assert "no-prices-at-all" not in table.prices
    assert table.resolve(None, "sample_spec") is None
    assert table.resolve("sample_spec", "sample_spec") is None


def test_resolve_prefers_the_provider_namespace_then_the_bare_model() -> None:
    table = PriceTable.from_payload(_PAYLOAD)
    assert table.resolve("deepseek", "deepseek-flash") is table.prices["deepseek/deepseek-flash"]
    # no `deepseek/deepseek-chat` in the payload: the bare key answers instead
    assert table.resolve("deepseek", "deepseek-chat") is table.prices["deepseek-chat"]
    assert table.resolve(None, "deepseek-chat") is table.prices["deepseek-chat"]


def test_resolve_is_exact_and_never_fuzzy() -> None:
    table = PriceTable.from_payload(_PAYLOAD)
    for provider, model in (
        ("deepseek", "deepseek-flas"),
        ("deepseek", "DeepSeek-Flash"),
        ("deepseek/x", "deepseek-flash"),
        ("deepseek", ""),
    ):
        assert table.resolve(provider, model) is None


def test_resolve_target_prefers_the_targets_table() -> None:
    table = PriceTable.from_payload(_PAYLOAD)
    found = resolve_target(_target(), "deepseek-flash", table)
    assert found == (_OWN, "table")


def test_resolve_target_falls_back_to_litellm() -> None:
    table = PriceTable.from_payload(_PAYLOAD)
    target = _target(prices={}, model="deepseek-flash")
    assert resolve_target(target, "deepseek-flash", table) == (
        table.prices["deepseek/deepseek-flash"],
        "litellm",
    )
    # a target that names no provider resolves the bare key only
    bare = _target(prices={}, model="deepseek-chat", provider=None)
    assert resolve_target(bare, "deepseek-chat", table) == (
        table.prices["deepseek-chat"],
        "litellm",
    )


def test_resolve_target_without_a_table_states_nothing() -> None:
    assert resolve_target(_target(prices={}), "deepseek-flash", None) is None
    table = PriceTable.from_payload(_PAYLOAD)
    assert resolve_target(_target(prices={}, model="absent"), "absent", table) is None


def test_load_table_reads_the_fixture() -> None:
    table = load_table(FIXTURE_TABLE)
    flash = table.prices["deepseek/deepseek-flash"]
    assert (flash.input, flash.output, flash.cache_read, flash.cache_write) == (
        pytest.approx(0.30),
        pytest.approx(1.20),
        pytest.approx(0.006),
        0.0,
    )
    sonnet = table.prices["claude-sonnet-4-5"]
    assert sonnet.cache_write == pytest.approx(3.75)
    assert "sample_spec" not in table.prices


def test_load_table_rejects_a_file_that_is_not_the_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No such file"):
        load_table(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_table(broken)
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_table(listed)


def test_cache_age_is_none_without_a_file_and_the_mtime_with_one(tmp_path: Path) -> None:
    path = tmp_path / "litellm-prices.json"
    assert cache_age_s(path, 1_000.0) is None
    path.write_text("{}", encoding="utf-8")
    assert cache_age_s(path, path.stat().st_mtime + 42.0) == pytest.approx(42.0)
    # a copy stamped in the future is not a negative age
    assert cache_age_s(path, path.stat().st_mtime - 42.0) == 0.0


def _fetching(payload: dict[str, Any], calls: list[int]) -> Any:
    """A fetcher that records its calls, for the tests that expect one."""

    async def fetch(client: object) -> dict[str, Any]:
        calls.append(1)
        return payload

    return fetch


def _never_fetch(calls: list[int]) -> Any:
    """A fetcher that fails the test if it is called at all."""

    async def fetch(client: object) -> dict[str, Any]:
        calls.append(1)
        raise AssertionError("the cache was fresh; nothing should have been fetched")

    return fetch


def _write_cache(path: Path, payload: dict[str, Any], *, age_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    stamp = path.stat().st_mtime - age_s
    os.utime(path, (stamp, stamp))


def test_a_fresh_copy_is_read_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "litellm-prices.json"
    _write_cache(path, _PAYLOAD, age_s=MAX_AGE_S - 1)
    calls: list[int] = []
    monkeypatch.setattr("provibench.bench.prices.fetch_payload", _never_fetch(calls))

    state = cached_table(path, update=False)
    assert state.fetched is False
    assert state.warning is None
    assert state.table.prices["deepseek/deepseek-flash"].input == pytest.approx(0.30)
    assert calls == []


def test_a_stale_copy_is_refetched_and_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "litellm-prices.json"
    _write_cache(path, {"old": {"input_cost_per_token": 1.0}}, age_s=MAX_AGE_S + 60)
    calls: list[int] = []
    monkeypatch.setattr("provibench.bench.prices.fetch_payload", _fetching(_PAYLOAD, calls))

    state = cached_table(path, update=False)
    assert state.fetched is True
    assert calls == [1]
    assert "deepseek/deepseek-flash" in json.loads(path.read_text(encoding="utf-8"))
    assert state.table.prices["deepseek/deepseek-flash"].input == pytest.approx(0.30)


def test_update_refetches_a_fresh_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "litellm-prices.json"
    _write_cache(path, {}, age_s=0.0)
    calls: list[int] = []
    monkeypatch.setattr("provibench.bench.prices.fetch_payload", _fetching(_PAYLOAD, calls))

    state = cached_table(path, update=True)
    assert state.fetched is True
    assert calls == [1]
    assert state.age_s == pytest.approx(0.0, abs=60.0)


def test_a_failed_refresh_falls_back_to_the_stale_copy(tmp_path: Path) -> None:
    path = tmp_path / "litellm-prices.json"
    _write_cache(path, _PAYLOAD, age_s=10 * 86_400)
    # the session-wide fixture's fetcher fails: no `monkeypatch` here on purpose
    state = cached_table(path, update=True)
    assert state.fetched is False
    assert state.age_s == pytest.approx(10 * 86_400, abs=60.0)
    assert state.table.prices["deepseek/deepseek-flash"].input == pytest.approx(0.30)
    assert state.warning is not None
    assert "could not be refreshed" in state.warning
    assert state.warning.endswith("using the copy fetched 10 days ago")


def test_a_failed_refresh_with_no_copy_is_table_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "litellm-prices.json"
    with pytest.raises(TableUnavailable) as caught:
        cached_table(path, update=False)
    assert "raw.githubusercontent.com" in str(caught.value)
    assert "targets.toml" in caught.value.hint
    assert str(path) in caught.value.hint
    assert "n/a" in caught.value.line


def test_a_corrupt_copy_is_no_copy(tmp_path: Path) -> None:
    path = tmp_path / "litellm-prices.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(TableUnavailable) as caught:
        cached_table(path, update=False)
    assert "not a usable table" in str(caught.value)


def _result(input_tokens: int = 1_000_000) -> ReplayResult:
    return ReplayResult(
        seq=1,
        turn=1,
        status=200,
        latency_ms=1.0,
        message_id="m1",
        prompt_total=input_tokens,
        cached=0,
        cache_write=0,
        output_tokens=0,
        usage=Usage(input_tokens=input_tokens),
    )


def test_enrich_anthropic_costs_from_the_community_table() -> None:
    """The breakdown a finished run records names the source the estimate showed."""
    table = PriceTable.from_payload(_PAYLOAD)
    result = _result()
    enrich_anthropic(RunSpec(target=_target(prices={}), model="deepseek-flash"), [result], table)
    assert result.cost is not None
    assert result.cost.source == "litellm"
    assert result.cost.input == pytest.approx(0.30)

    overridden = _result()
    enrich_anthropic(RunSpec(target=_target(), model="deepseek-flash"), [overridden], table)
    assert overridden.cost is not None
    assert overridden.cost.source == "table"
    assert overridden.cost.input == pytest.approx(0.15)

    unpriced = _result()
    spec = RunSpec(target=_target(prices={}, model="absent"), model="absent")
    enrich_anthropic(spec, [unpriced], table)
    assert unpriced.cost is None
    assert unpriced.note == "no price table for absent"


def test_format_age() -> None:
    assert format_age(3.0) == "less than a minute ago"
    assert format_age(90.0) == "1 minute ago"
    assert format_age(12 * 60.0) == "12 minutes ago"
    assert format_age(3600.0) == "1 hour ago"
    assert format_age(5 * 3600.0) == "5 hours ago"
    assert format_age(24 * 3600.0) == "1 day ago"
    assert format_age(3 * 86_400.0 + 60) == "3 days ago"
