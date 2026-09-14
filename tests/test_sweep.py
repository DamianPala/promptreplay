"""`sweep`: the endpoint expansion, the filters, the ordering, and the parallel fan-out.

The only boundary mocked is the network `httpx.AsyncClient` opens, so these tests exercise
the command, the spec building, the probe protocol and the aggregation together: the fake
transport serves the endpoint list, the `/generation` lookup and the probe requests.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from provibench.bench.trace import RecordedResponse, TraceEntry, Usage, append_entry
from tests.conftest import BenchPaths, Cli

_RealAsyncClient = httpx.AsyncClient

_MODEL = "deepseek/deepseek-v4.1-flash"
_NATIVE = "deepseek-flash"

_TARGETS_TOML = (
    "[targets.or]\n"
    'url = "https://openrouter.test/v1/messages"\n'
    'api_key_env = "OR_KEY"\n'
    'kind = "openrouter"\n'
    "\n"
    "[targets.deepseek]\n"
    'url = "https://api.deepseek.test/anthropic/v1/messages"\n'
    'api_key_env = "DS_KEY"\n'
    'kind = "anthropic"\n'
    "\n"
    "[targets.deepseek.aliases]\n"
    f'"{_MODEL}" = "{_NATIVE}"\n'
    "\n"
    f'[targets.deepseek.prices."{_NATIVE}"]\n'
    "input = 0.15\n"
    "cache_read = 0.003\n"
    "cache_write = 0.15\n"
    "output = 0.60\n"
)
_ALIAS_LINE = f'"{_MODEL}" = "{_NATIVE}"\n'

# USD per M tokens as OpenRouter reports them: input, output, cache read.
_PRICING: dict[str, dict[str, str]] = {
    "novita": {"prompt": "0.0000003", "completion": "0.0000006", "input_cache_read": "0.00000003"},
    "novita/fp8": {
        "prompt": "0.0000004",
        "completion": "0.0000008",
        "input_cache_read": "0",
    },
    "siliconflow": {"prompt": "0.0000002", "completion": "0.0000004", "input_cache_read": "0"},
    "gmicloud": {
        "prompt": "0.00000032",
        "completion": "0.0000006",
        "input_cache_read": "0.000000032",
    },
}
_ENV = {"OR_KEY": "secret", "DS_KEY": "secret"}


def _endpoint(tag: str) -> dict[str, Any]:
    return {
        "provider_name": tag.split("/")[0].title(),
        "tag": tag,
        "quantization": "fp8",
        "context_length": 128000,
        "pricing": _PRICING[tag],
    }


_ENDPOINTS = [
    _endpoint("novita"),
    _endpoint("novita/fp8"),
    _endpoint("siliconflow"),
    _endpoint("gmicloud"),
]


def _write_targets(path: Path, *, aliases: bool = True) -> None:
    toml = _TARGETS_TOML if aliases else _TARGETS_TOML.replace(_ALIAS_LINE, "")
    path.write_text(toml)


def _trace_entry(seq: int, prompt_tokens: int) -> TraceEntry:
    return TraceEntry(
        seq=seq,
        ts=f"2026-01-01T00:00:{seq:02d}Z",
        path="/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        body={
            "model": "orig",
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{"role": "user", "content": f"question {seq}"}],
            "max_tokens": 1024,
        },
        conversation="c1",
        response=RecordedResponse(
            status=200, latency_ms=1.0, usage=Usage(input_tokens=prompt_tokens)
        ),
    )


def _write_trace(path: Path, prompts: tuple[int, ...] = (100, 200, 300)) -> None:
    for seq, prompt_tokens in enumerate(prompts, start=1):
        append_entry(path, _trace_entry(seq, prompt_tokens))


def _provider(body: dict[str, Any]) -> dict[str, Any]:
    """The `provider` block a request body carries; empty when it pins no endpoint."""
    return cast("dict[str, Any]", body.get("provider") or {})


def _pinned(body: dict[str, Any]) -> list[str]:
    """The endpoint tags a request pins, in the order it names them."""
    only: object = _provider(body).get("only")
    if not isinstance(only, list):
        return []
    return [str(name) for name in cast("list[object]", only)]


def _spec_key(body: dict[str, Any]) -> str:
    """Which spec a request belongs to: its pinned tag, else the model it sends."""
    tags = _pinned(body)
    return tags[0] if tags else str(body.get("model"))


def _turn(body: dict[str, Any]) -> int:
    """Which recorded turn a probe request sends, read off its last user message."""
    return int(str(body["messages"][-1]["content"]).split()[-1])


def _stream(cached: int, prompt: int) -> httpx.Response:
    """A streamed generation: one text delta, and the usage the summary folds."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "model": "stream-model",
                "usage": {
                    "input_tokens": prompt - cached,
                    "cache_read_input_tokens": cached,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200,
        content=("\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\n").encode(),
        headers={"content-type": "text/event-stream"},
    )


def _transport(
    *,
    cached: Mapping[str, int] | None = None,
    endpoints: list[dict[str, Any]] | None = None,
    sent: list[dict[str, Any]] | None = None,
    order: list[tuple[str, int]] | None = None,
    calls: list[str] | None = None,
    models: Mapping[str, str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """The fake provider: endpoint list, generation lookup, and the probe requests.

    `cached` is the cached-token count a spec's reads report, keyed by its pinned tag or,
    for a native spec, by the model it sends; a spec's first request — its cold write —
    always reports zero, the way a fresh nonce makes it. Every message body is appended to
    `sent`, every request's `(spec, turn)` to `order`, and every request's path to `calls`,
    which is what a test asserting on the order requests went out in reads.
    """
    hits = dict(cached or {})
    answers = dict(models or {})
    seen: dict[str, int] = {}

    def serve(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(
                200, json={"data": {"endpoints": _ENDPOINTS if endpoints is None else endpoints}}
            )
        if request.url.path.endswith("/generation"):
            return httpx.Response(200, json={"data": {"id": "g1", "total_cost": 0.0001}})
        body = json.loads(request.content)
        key = _spec_key(body)
        if sent is not None:
            sent.append(body)
        if order is not None:
            order.append((key, _turn(body)))
        count = seen.get(key, 0)
        seen[key] = count + 1
        hit = 0 if count == 0 else hits.get(key, 0)
        prompt = 100
        if body.get("stream"):
            return _stream(hit, prompt)
        # the endpoint answers with its own model name, the gateway with its own slug
        gateway = "openrouter" in request.url.host
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "model": answers.get(key) or (_MODEL if gateway else str(body.get("model"))),
                "usage": {
                    "input_tokens": prompt - hit,
                    "cache_read_input_tokens": hit,
                    "output_tokens": 1,
                },
            },
        )

    return serve


def _install(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _yielding(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[..., Any]:
    """A transport handler that yields to the event loop, so parallel specs interleave."""

    async def serve(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)
        return handler(request)

    return serve


def _sweep(cli: Cli, bench_paths: BenchPaths, *args: str) -> Any:
    trace = bench_paths.traces_dir / "t.jsonl"
    if not trace.exists():
        _write_trace(trace)
    return cli.run("sweep", "t", _MODEL, *args, env={**bench_paths.env, **_ENV})


def _labels(summaries: Any) -> list[str]:
    """The label of every summary, in the order the document carries them."""
    return [str(summary["label"]) for summary in summaries]


def _one_rung(*extra: str) -> tuple[str, ...]:
    """Flags that keep a test to one rung, one warm read, no waiting and no generation."""
    return ("--rungs", "1", "--repeats", "1", "--gap", "0", "--no-throughput", "--yes", *extra)


def test_sweep_expands_one_pinned_spec_per_endpoint(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(cached={"novita": 90, _NATIVE: 100}, sent=sent))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr
    assert set(_labels(outcome.document["summaries"])) == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@novita/fp8",
        f"or:{_MODEL}@siliconflow",
        f"or:{_MODEL}@gmicloud",
        f"deepseek:{_NATIVE}",
    }
    pinned = {tags[0] for body in sent if (tags := _pinned(body))}
    assert pinned == {"novita", "novita/fp8", "siliconflow", "gmicloud"}
    # every pinned request tells the gateway not to fall back to another endpoint
    assert all(_provider(body).get("allow_fallbacks") is False for body in sent if _pinned(body))


def test_sweep_include_and_exclude_filter_by_tag_prefix(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(cached={_NATIVE: 100}))

    # a prefix keeps `novita/fp8` too, and the native spec is not an endpoint to filter
    included = _sweep(cli, bench_paths, *_one_rung("--include", "novita"))
    assert included.code == 0, included.stderr
    assert set(_labels(included.document["summaries"])) == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@novita/fp8",
        f"deepseek:{_NATIVE}",
    }

    both = _sweep(cli, bench_paths, *_one_rung("--include", "novita", "--exclude", "novita/fp8"))
    assert both.code == 0, both.stderr
    assert set(_labels(both.document["summaries"])) == {
        f"or:{_MODEL}@novita",
        f"deepseek:{_NATIVE}",
    }


def test_sweep_unknown_tag_names_the_available_ones(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(sent=sent))

    outcome = _sweep(cli, bench_paths, *_one_rung("--include", "novitaa"))
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    message = str(outcome.error["message"])
    assert "novitaa" in message
    assert "novita/fp8" in message and "siliconflow" in message and "gmicloud" in message
    assert sent == []


def test_sweep_an_empty_endpoint_list_after_filtering_is_an_input_error(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(sent=sent))

    outcome = _sweep(
        cli,
        bench_paths,
        *_one_rung("--exclude", "novita", "--exclude", "siliconflow", "--exclude", "gmicloud"),
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "empty after filtering" in str(outcome.error["message"])
    assert sent == []


def test_sweep_adds_a_native_spec_through_the_alias(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(sent=sent))

    without = _sweep(cli, bench_paths, *_one_rung("--include", "novita", "--exclude", "novita/fp8"))
    assert without.code == 0, without.stderr
    assert set(_labels(without.document["summaries"])) == {f"or:{_MODEL}@novita"}

    _write_targets(bench_paths.targets_path)
    sent.clear()
    with_alias = _sweep(
        cli, bench_paths, *_one_rung("--include", "novita", "--exclude", "novita/fp8")
    )
    assert with_alias.code == 0, with_alias.stderr
    assert set(_labels(with_alias.document["summaries"])) == {
        f"or:{_MODEL}@novita",
        f"deepseek:{_NATIVE}",
    }
    native = next(s for s in with_alias.document["summaries"] if s["price_source"] == "table")
    assert native["label"] == f"deepseek:{_NATIVE}"
    # the native spec sends the name its own endpoint serves, not the OpenRouter slug
    assert any(body.get("model") == _NATIVE for body in sent)


def test_sweep_records_the_sweep_block_and_the_endpoint_snapshot(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport())

    outcome = _sweep(cli, bench_paths, *_one_rung("--include", "novita", "--exclude", "novita/fp8"))
    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["protocol"] == "probe"
    assert meta["sweep"] == {
        "model": _MODEL,
        "target": "or",
        "included": ["novita"],
        "excluded": ["novita/fp8"],
    }
    assert meta["endpoints"][_MODEL]  # the snapshot the estimate was priced from
    assert [spec["label"] for spec in meta["specs"]] == _labels(outcome.document["summaries"])
    assert outcome.document["sweep"] == meta["sweep"]


def test_sweep_orders_by_effective_price(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured hit rate sets the order: siliconflow is the cheapest to list, the
    dearest to run, and `@novita/fp8` caches nothing at all."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(cached={"novita": 90, "gmicloud": 90, _NATIVE: 100}))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr
    effective = [float(summary["eff_per_m_prompt"]) for summary in outcome.document["summaries"]]
    assert effective == sorted(effective)
    assert _labels(outcome.document["summaries"]) == [
        f"deepseek:{_NATIVE}",  # 0.003
        f"or:{_MODEL}@novita",  # (1-.9)*0.30 + .9*0.03
        f"or:{_MODEL}@gmicloud",  # (1-.9)*0.32 + .9*0.032
        f"or:{_MODEL}@siliconflow",  # nothing cached: the full 0.20
        f"or:{_MODEL}@novita/fp8",  # nothing cached, and the priciest endpoint
    ]


def test_report_of_a_sweep_run_reads_the_same_order_with_no_network(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run directory keeps the measured order, so `report` needs no endpoint list."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(cached={"novita": 90, "gmicloud": 90, _NATIVE: 100}))
    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"report went to the network: {request.url}")

    _install(monkeypatch, refuse)
    report = cli.run("report", str(outcome.document["run_dir"]), env={**bench_paths.env, **_ENV})
    assert report.code == 0, report.stderr
    assert _labels(report.document["probe_summaries"]) == _labels(outcome.document["summaries"])


def test_sweep_ties_are_broken_by_hit_rate(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    # neither endpoint discounts the cache read, so both cost the same per prompt token
    # however they cache; the one that caches more of the prefix is the one to use
    flat = {"prompt": "0.0000005", "completion": "0.0000005", "input_cache_read": "0.0000005"}
    endpoints = [_endpoint("siliconflow"), _endpoint("gmicloud")]
    for endpoint in endpoints:
        endpoint["pricing"] = flat
    _install(monkeypatch, _transport(endpoints=endpoints, cached={"siliconflow": 100}))

    outcome = _sweep(
        cli, bench_paths, *_one_rung("--include", "siliconflow", "--include", "gmicloud")
    )
    assert outcome.code == 0, outcome.stderr
    assert _labels(outcome.document["summaries"]) == [
        f"or:{_MODEL}@siliconflow",
        f"or:{_MODEL}@gmicloud",
    ]
    assert [summary["hit_rate"] for summary in outcome.document["summaries"]] == [1.0, 0.0]


def test_sweep_parallel_sends_different_specs_at_once_and_keeps_each_spec_in_order(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    order: list[tuple[str, int]] = []
    _install(
        monkeypatch,
        _yielding(_transport(order=order, cached={"novita": 90, "gmicloud": 90})),
    )

    outcome = _sweep(cli, bench_paths, *_one_rung(), "--parallel", "3")
    assert outcome.code == 0, outcome.stderr
    tags = {key for key, _ in order}
    assert len(tags) == 5  # four endpoints plus the native spec the alias adds
    assert len({key for key, _ in order[:3]}) == 3  # three specs in flight, one each
    for tag in tags:
        # cold turn 1, then the warm read of turn 2: one spec never overtakes itself
        assert [turn for key, turn in order if key == tag] == [1, 2]
    assert len(order) == 10


def test_sweep_parallel_one_runs_one_spec_at_a_time(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    order: list[tuple[str, int]] = []
    _install(monkeypatch, _yielding(_transport(order=order, cached={"novita": 90})))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr
    keys = [key for key, _ in order]
    pairs = [keys[index : index + 2] for index in range(0, len(keys), 2)]
    assert all(pair[0] == pair[1] for pair in pairs)  # no spec ever overlaps another
    assert len({pair[0] for pair in pairs}) == 5


def test_sweep_runs_the_whole_probe_protocol_including_the_stream_and_the_ttl_read(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every `probe` flag applies: the throughput request and `--ttl` ride along."""
    _write_targets(bench_paths.targets_path, aliases=False)

    async def fake_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(cached={"novita": 90}, sent=sent))

    outcome = _sweep(
        cli, bench_paths, "--rungs", "1", "--repeats", "2", "--gap", "0", "--ttl", "5", "--yes"
    )
    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    [spec] = [ref for ref in meta["specs"] if ref["label"].endswith("@novita")]
    records = [
        json.loads(line)
        for line in (run_dir / str(spec["file"])).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["role"] for record in records] == [
        "cold",
        "warm",
        "warm",
        "stream",
        "ttl",
    ]
    assert [record["cached"] for record in records] == [0, 90, 90, 90, 90]
    assert meta["ttl_s"] == [5]
    [summary] = [
        summary for summary in outcome.document["summaries"] if summary["label"].endswith("@novita")
    ]
    assert summary["ttft_ms"] is not None
    assert summary["rungs"][0]["ttl"][0]["hit"] is True


def test_sweep_prints_the_probe_tables_in_the_measured_order(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _write_trace(bench_paths.traces_dir / "t.jsonl")
    _install(monkeypatch, _transport(cached={"novita": 90, "gmicloud": 90, _NATIVE: 100}))

    outcome = cli.run(
        "sweep",
        "t",
        _MODEL,
        *_one_rung(),
        tty_stdout=True,
        env={**bench_paths.env, **_ENV},
    )
    assert outcome.code == 0, outcome.stderr
    assert "hit %" in outcome.stdout and "cached cold" in outcome.stdout
    spec_rows = [
        line
        for line in outcome.stdout.splitlines()
        if line.startswith(("@", "deepseek:")) and "|" in line
    ]
    assert [row.split("|")[0].strip() for row in spec_rows[:5]] == [
        "deepseek:deepseek-flash",
        "@novita",
        "@gmicloud",
        "@siliconflow",
        "@novita/fp8",
    ]
    assert all(len(line) <= 120 for line in outcome.stdout.splitlines())


def test_sweep_estimates_every_spec_and_the_budget_refuses_above_it(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(sent=sent))

    refused = _sweep(cli, bench_paths, *_one_rung("--budget", "0.00001"))
    assert refused.code == 2
    assert refused.error["kind"] == "invalid_input"
    assert "exceeds --budget" in str(refused.error["message"])
    assert sent == []
    assert not (bench_paths.runs_dir / "t").exists()

    within = _sweep(cli, bench_paths, *_one_rung("--budget", "0.01"))
    assert within.code == 0, within.stderr
    # one estimate row per spec, named by the same column the tables use: the shared
    # `target:model` is in the caption and each row keeps the `@tag` a reader picks with
    estimate = within.stderr
    assert f"specs: or:{_MODEL}@<provider>" in estimate
    for tag in ("@novita", "@novita/fp8", "@siliconflow", "@gmicloud"):
        assert tag in estimate
    assert f"deepseek:{_NATIVE}" in estimate  # the native spec keeps its whole label
    assert estimate.count("openrouter-endpoint") == 4  # the gateway specs
    assert "table" in estimate  # the native spec is priced from targets.toml


def test_sweep_row_labels_are_the_run_specs_probe_takes_by_hand(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row label is a run spec: it parses back to the endpoint the sweep pinned, slash
    tags included, which is what makes a sweep row copy-pasteable into `probe`."""
    from provibench.bench.targets import load_targets, parse_run_spec

    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport())
    outcome = _sweep(cli, bench_paths, *_one_rung("--include", "novita"))
    assert outcome.code == 0, outcome.stderr

    targets = load_targets(bench_paths.targets_path)
    for label in _labels(outcome.document["summaries"]):
        assert parse_run_spec(label, targets).label == label
    pinned = parse_run_spec(f"or:{_MODEL}@novita/fp8", targets)
    assert (pinned.model, pinned.providers) == (_MODEL, ["novita/fp8"])
    # the persisted spec of that row pins the same endpoint the label names
    meta = json.loads((Path(str(outcome.document["run_dir"])) / "run.json").read_text("utf-8"))
    for ref in meta["specs"]:
        assert parse_run_spec(str(ref["label"]), targets).providers == ref["providers"]


def test_sweep_resolves_the_trace_before_it_fetches_endpoints(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped TRACE is the cheapest mistake to make: it costs no round trip."""
    _write_targets(bench_paths.targets_path)
    calls: list[str] = []
    _install(monkeypatch, _transport(calls=calls))

    outcome = cli.run("sweep", "typo", _MODEL, *_one_rung(), env={**bench_paths.env, **_ENV})
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"
    assert "typo" in str(outcome.error["message"])
    assert calls == []


def test_sweep_records_how_parallel_the_run_was(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Latency medians of a contended run are not comparable with a sequential one."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport())

    concurrent = _sweep(cli, bench_paths, *_one_rung("--parallel", "3"))
    assert concurrent.code == 0, concurrent.stderr
    meta = json.loads((Path(str(concurrent.document["run_dir"])) / "run.json").read_text("utf-8"))
    assert any(note.startswith("parallel 3") for note in meta["notes"])

    sequential = _sweep(cli, bench_paths, *_one_rung())
    assert sequential.code == 0, sequential.stderr
    meta = json.loads((Path(str(sequential.document["run_dir"])) / "run.json").read_text("utf-8"))
    assert not any("parallel" in note for note in meta["notes"])


def test_sweep_reports_endpoints_that_came_back_without_a_tag(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    endpoints = [_endpoint("novita"), {**_endpoint("siliconflow"), "tag": ""}]
    _install(monkeypatch, _transport(endpoints=endpoints))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr
    assert set(_labels(outcome.document["summaries"])) == {
        f"or:{_MODEL}@novita",
        f"deepseek:{_NATIVE}",
    }
    assert "1 endpoint(s)" in outcome.stderr and "without a tag" in outcome.stderr


def test_sweep_says_why_an_all_untagged_endpoint_list_is_empty(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filters are not what emptied it, and the message has to say what did."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(endpoints=[{**_endpoint("novita"), "tag": ""}]))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    message = str(outcome.error["message"])
    assert "empty after filtering" in message
    assert "1 of 1 came back without a tag" in message


def test_sweep_target_flag_picks_the_gateway_target(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport())

    unknown = _sweep(cli, bench_paths, *_one_rung("--target", "nope"))
    assert unknown.code == 2
    assert "known targets: deepseek, or" in str(unknown.error["message"])

    native = _sweep(cli, bench_paths, *_one_rung("--target", "deepseek"))
    assert native.code == 2
    assert native.error["kind"] == "invalid_input"
    assert 'kind="openrouter"' in str(native.error["message"])

    named = _sweep(cli, bench_paths, *_one_rung("--target", "or", "--include", "novita"))
    assert named.code == 0, named.stderr
    meta = json.loads((Path(str(named.document["run_dir"])) / "run.json").read_text("utf-8"))
    assert meta["sweep"]["target"] == "or"


def test_sweep_without_an_openrouter_target_is_an_input_error(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench_paths.targets_path.write_text(
        '[targets.deepseek]\nurl = "https://api.deepseek.test/v1/messages"\n'
        'api_key_env = "DS_KEY"\nkind = "anthropic"\n'
    )
    _install(monkeypatch, _transport())

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "openrouter" in str(outcome.error["message"])


def test_sweep_fails_when_the_model_has_no_endpoints(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(endpoints=[], sent=sent))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    assert _MODEL in str(outcome.error["message"])
    assert sent == []
