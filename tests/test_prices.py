"""`prices`, and the price a native spec is estimated and costed with.

The command is the table's own window onto the cache; the replay tests below are the run
side of the same wiring — the estimate before the run is paid for, and the breakdown after.
The only boundaries mocked are the LiteLLM fetch (`no_price_network` in conftest, overridden
per test where a fetch must succeed) and the httpx client the requests go out on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from provibench.bench.trace import (
    RecordedResponse,
    TraceEntry,
    Usage,
    append_entry,
)
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli, install_price_cache

_RealAsyncClient = httpx.AsyncClient

_TARGET = (
    "[targets.deepseek]\n"
    'url = "https://api.deepseek.test/anthropic/v1/messages"\n'
    'api_key_env = "DEEPSEEK_KEY"\n'
    'kind = "anthropic"\n'
    'litellm_provider = "deepseek"\n'
    "\n"
    "[targets.deepseek.aliases]\n"
    '"deepseek/deepseek-v4.1-flash" = "deepseek-flash"\n'
)

_OVERRIDE = (
    '[targets.deepseek.prices."deepseek-flash"]\n'
    "input = 0.15\n"
    "cache_read = 0.003\n"
    "cache_write = 0.15\n"
    "output = 0.60\n"
)

_ENV = {"DEEPSEEK_KEY": "secret"}

_FETCHED: dict[str, Any] = {
    "deepseek/deepseek-flash": {
        "input_cost_per_token": 9e-7,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 9e-9,
        "cache_creation_input_token_cost": 0.0,
        "litellm_provider": "deepseek",
        "mode": "chat",
    }
}


def _write_targets(path: Path, *, override: bool) -> None:
    path.write_text(_TARGET + (_OVERRIDE if override else ""), encoding="utf-8")


def _trace_entry(seq: int) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        body={
            "model": "orig",
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        },
        conversation="c1",
        response=RecordedResponse(status=200, latency_ms=1.0, usage=Usage(input_tokens=100)),
    )


def _write_trace(bench_paths: BenchPaths) -> None:
    append_entry(bench_paths.traces_dir / "t.jsonl", _trace_entry(1))


def _client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.AsyncClient]:
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)  # type: ignore[arg-type]

    return factory


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "model": "deepseek-flash",
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1,
            },
        },
    )


def _replay(cli: Cli, bench_paths: BenchPaths, *args: str) -> Any:
    return cli.run("replay", "t", *args, env={**bench_paths.env, **_ENV})


def _rows(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [entry for entry in map(as_document, as_list(document.get("prices")) or []) if entry]


def _row(document: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    return next(row for row in _rows(document) if row["model"] == model)


def _fetching(payload: dict[str, Any], calls: list[int]) -> Any:
    async def fetch(client: object) -> dict[str, Any]:
        calls.append(1)
        return payload

    return fetch


def _never_fetch(calls: list[int]) -> Any:
    async def fetch(client: object) -> dict[str, Any]:
        calls.append(1)
        raise AssertionError("nothing should have been fetched")

    return fetch


def test_prices_lists_the_native_targets_own_models_with_the_targets_source(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path, override=True)
    cache_path = install_price_cache(cli)

    outcome = cli.run("prices", "--json", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    document = outcome.document
    assert set(document) == {
        "cache_path",
        "source_url",
        "cache_age_s",
        "cache_age",
        "fetched",
        "note",
        "prices",
        "changed",
    }
    assert document["cache_path"] == str(cache_path)
    assert document["fetched"] is False and document["changed"] is False
    assert document["cache_age"] == "less than a minute ago"
    assert document["note"] is None
    assert "litellm" in str(document["source_url"])
    # the target names one model: the key of its prices table, which its alias also maps to
    assert [row["model"] for row in _rows(document)] == ["deepseek-flash"]
    row = _row(document, "deepseek-flash")
    assert row["target"] == "deepseek" and row["source"] == "targets"
    assert (row["input"], row["cache_read"], row["cache_write"], row["output"]) == (
        0.15,
        0.003,
        0.15,
        0.6,
    )


def test_prices_keeps_model_name_whole_at_eighty_columns(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, override=True)
    install_price_cache(cli)
    monkeypatch.setenv("COLUMNS", "80")

    outcome = cli.run("prices", "deepseek-flash", "deepseek-chat", tty=True, env=bench_paths.env)

    assert outcome.code == 0, outcome.stderr
    assert "deepseek-flash" in outcome.stdout


def test_prices_prices_a_model_from_litellm_when_the_override_is_gone(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path, override=False)
    install_price_cache(cli)

    outcome = cli.run("prices", "--json", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    row = _row(outcome.document, "deepseek-flash")
    assert row["source"] == "litellm"
    assert row["input"] == pytest.approx(0.30)
    assert row["cache_read"] == pytest.approx(0.006)
    assert row["cache_write"] == 0.0
    assert row["output"] == pytest.approx(1.20)


def test_prices_names_a_model_nothing_prices(cli: Cli, bench_paths: BenchPaths) -> None:
    _write_targets(bench_paths.targets_path, override=True)
    install_price_cache(cli)

    outcome = cli.run("prices", "--json", "deepseek-flash", "absent-model", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert [row["model"] for row in _rows(outcome.document)] == ["deepseek-flash", "absent-model"]
    row = _row(outcome.document, "absent-model")
    assert row["source"] == "n/a"
    assert all(row[field] is None for field in ("input", "cache_read", "cache_write", "output"))


def test_prices_update_fetches_once_and_rewrites_the_cache(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, override=False)
    calls: list[int] = []
    monkeypatch.setattr("provibench.bench.prices.fetch_payload", _fetching(_FETCHED, calls))
    cache = cli.home / ".cache" / "provibench" / "litellm-prices.json"
    assert not cache.exists()

    outcome = cli.run("prices", "--update", "--json", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert calls == [1]
    assert json.loads(cache.read_text(encoding="utf-8")) == _FETCHED
    document = outcome.document
    assert document["fetched"] is True and document["changed"] is True
    row = _row(document, "deepseek-flash")
    assert (row["source"], row["input"]) == ("litellm", pytest.approx(0.90))

    # the copy is fresh now: a second read serves the fetched price without another call
    again = cli.run("prices", "--json", env=bench_paths.env)
    assert again.code == 0, again.stderr
    assert calls == [1]
    assert again.document["fetched"] is False
    assert _row(again.document, "deepseek-flash")["input"] == pytest.approx(0.90)


def test_prices_reads_a_fresh_cache_without_fetching(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, override=False)
    install_price_cache(cli)
    calls: list[int] = []
    monkeypatch.setattr("provibench.bench.prices.fetch_payload", _never_fetch(calls))

    outcome = cli.run("prices", "--json", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert calls == []
    assert _row(outcome.document, "deepseek-flash")["input"] == pytest.approx(0.30)


def test_a_stale_cache_and_a_failing_fetcher_warn_and_keep_the_prices(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path, override=False)
    install_price_cache(cli, age_s=8 * 86_400)

    outcome = cli.run("prices", "--json", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    document = outcome.document
    assert document["fetched"] is False and document["changed"] is False
    assert document["cache_age"] == "8 days ago"
    assert "could not be refreshed" in str(document["note"])
    assert "8 days ago" in str(document["note"])
    assert _row(document, "deepseek-flash")["input"] == pytest.approx(0.30)


def test_no_cache_and_a_failing_fetcher_is_operation_failed(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_targets(bench_paths.targets_path, override=False)

    outcome = cli.run("prices", "--json", env=bench_paths.env)
    assert outcome.code == 1
    error = outcome.error
    assert error["kind"] == "operation_failed"
    assert "raw.githubusercontent.com" in str(error["message"])
    assert "targets.toml" in str(error["hint"])
    assert str(cli.home / ".cache") in str(error["hint"])


def test_a_native_estimate_prices_from_the_override(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, override=True)
    _write_trace(bench_paths)
    install_price_cache(cli)
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))

    outcome = _replay(cli, bench_paths, "--run", "deepseek:deepseek-flash", "--yes")
    assert outcome.code == 0, outcome.stderr
    assert "0.150" in outcome.stderr and "targets" in outcome.stderr
    assert "litellm" not in outcome.stderr


def test_a_native_estimate_prices_from_litellm_without_the_override(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, override=False)
    _write_trace(bench_paths)
    install_price_cache(cli)
    calls: list[int] = []
    monkeypatch.setattr("provibench.bench.prices.fetch_payload", _never_fetch(calls))
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))

    outcome = _replay(cli, bench_paths, "--run", "deepseek:deepseek-flash", "--yes")
    assert outcome.code == 0, outcome.stderr
    # a fresh copy is read: a run never fetches a table it already has
    assert calls == []
    assert "0.300" in outcome.stderr and "litellm" in outcome.stderr
    # the peak-rate caveat travels with the estimate, not only with the README
    assert "peak rates" in outcome.stderr


def test_a_stale_cache_still_prices_a_run_and_warns(cli: Cli, bench_paths: BenchPaths) -> None:
    """The offline week between refreshes: the copy on disk is old, and it is still the price."""
    _write_targets(bench_paths.targets_path, override=False)
    _write_trace(bench_paths)
    install_price_cache(cli, age_s=8 * 86_400)

    outcome = _replay(cli, bench_paths, "--run", "deepseek:deepseek-flash", "--budget", "1")
    assert outcome.code == 2
    assert "could not be refreshed" in outcome.stderr
    assert "using the copy fetched 8 days ago" in outcome.stderr
    assert "0.300" in outcome.stderr and "litellm" in outcome.stderr


def test_a_model_the_table_does_not_list_is_unpriced_end_to_end(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """A present table with no row for the model: `n/a`, and `--budget` refuses as before."""
    _write_targets(bench_paths.targets_path, override=False)
    _write_trace(bench_paths)
    install_price_cache(cli)

    outcome = _replay(cli, bench_paths, "--run", "deepseek:unlisted-model", "--budget", "1")
    assert outcome.code == 2
    assert "no LiteLLM price table" not in outcome.stderr
    error = outcome.error
    assert error["kind"] == "invalid_input"
    assert "no listed price for deepseek:unlisted-model" in str(error["message"])


def test_a_run_without_a_table_degrades_to_n_a_and_the_budget_refuses(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """No cache and no reachable table: the run is priced as it was before the table existed."""
    _write_targets(bench_paths.targets_path, override=False)
    _write_trace(bench_paths)

    outcome = _replay(cli, bench_paths, "--run", "deepseek:deepseek-flash", "--budget", "1")
    assert outcome.code == 2
    assert "no LiteLLM price table at https://" in outcome.stderr
    assert "native prices stay n/a" in outcome.stderr
    error = outcome.error
    assert error["kind"] == "invalid_input"
    assert "no listed price for deepseek:deepseek-flash" in str(error["message"])
