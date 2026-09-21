"""`sweep`: the endpoint expansion, the filters, selection criteria, ordering, fan-out.

The only boundary mocked is the network `httpx.AsyncClient` opens, so these tests exercise
the command, the spec building, the probe protocol and the aggregation together: the fake
transport serves the endpoint list, the ZDR list, the `/generation` lookup, the probe's
requests and the availability pre-check, which it tells from a probe request the way the
run does — a probe request carries the probe's nonce, the pre-check does not.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from provibench.bench.selection import SweepInfo
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage, append_entry
from provibench.core.documents import as_document, as_list
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
    "relace/fp4": {"prompt": "0.0000009", "completion": "0.000001", "input_cache_read": "0"},
    "parasail/fp8": {
        "prompt": "0.00000028",
        "completion": "0.00000056",
        "input_cache_read": "0.000000028",
    },
    "openinference": {"prompt": "0.0000001", "completion": "0.0000002", "input_cache_read": "0"},
}
_ENV = {"OR_KEY": "secret", "DS_KEY": "secret"}

# The account-level exclusion OpenRouter answers a pinned request with when the settings of
# the key rule every candidate out; the reason line is what the run has to quote.
_GUARDRAIL_404: dict[str, Any] = {
    "type": "not_found_error",
    "message": "0 endpoints out of 1 requested are available matching your guardrail "
    "restrictions and data policy. We removed them for the following reasons (an endpoint "
    "may have matched multiple reasons):\nPaid model training violation (account settings): "
    "1 endpoint excluded; configurable at https://openrouter.ai/settings/privacy",
    "error_type": "not_found",
}
_GUARDRAIL_REASON = (
    "unavailable for this key: Paid model training violation (account settings) "
    "(change this at https://openrouter.ai/settings/privacy: allow this provider to train "
    "on prompts, or drop the training-data restriction)"
)

# The 429 the live sweep saw: the gateway wraps the provider's refusal in its own envelope,
# whose `type` is the word `error`; only the object it wraps names the failure.
_RATE_LIMIT_429: dict[str, Any] = {
    "type": "error",
    "error": {
        "type": "rate_limit_error",
        "message": "Provider returned error",
        "error_type": "rate_limit_exceeded",
    },
}
_RATE_LIMIT_REASON = "unavailable: rate_limit_exceeded"


def _endpoint(tag: str, **extra: object) -> dict[str, Any]:
    """One endpoint of the fake list: the shared prices plus whatever health it is given."""
    return {
        "provider_name": tag.split("/")[0].title(),
        "tag": tag,
        "quantization": "fp8",
        "context_length": 128000,
        "pricing": _PRICING[tag],
        **extra,
    }


def _health(
    uptime: float,
    *,
    status: int = 0,
    latency: float | None = None,
    throughput: float | None = None,
) -> dict[str, Any]:
    """The health fields of one endpoint; a percentile left out is one the API never sent."""
    return {
        "status": status,
        "uptime_last_1d": uptime,
        "latency_last_30m": latency,
        "throughput_last_30m": throughput,
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


class _Provider:
    """The fake OpenRouter: endpoint list, ZDR list, generation lookup, probes, pre-checks.

    `cached` is the cached-token count a spec's reads report, keyed by its pinned tag or,
    for a native spec, by the model it sends; a spec's first request — its cold write —
    always reports zero, the way a fresh nonce makes it. `unavailable` answers any request
    for that tag — the availability check's or the probe's — with the response it maps to:
    an account-level exclusion for a candidate, a failing endpoint for a spec the run
    nevertheless paid for; `limited` does the same for the 429 the gateway wraps in its own
    envelope. Every message body is appended to `sent`, every probe request's `(spec, turn)`
    to `order`, and every request's path to `calls`, which is what a test asserting on the
    order requests went out in reads.

    A pre-check is a request without the probe's nonce. It is recorded on `prechecked` and
    leaves the warm-up counter alone: it is a separate request rather than the cold write of
    the rung whose body it borrows.
    """

    def __init__(  # noqa: PLR0913 (one keyword per part of the fake a test varies)
        self,
        *,
        cached: Mapping[str, int] | None = None,
        endpoints: list[dict[str, Any]] | None = None,
        zdr: list[dict[str, Any]] | None = None,
        unavailable: Mapping[str, dict[str, Any]] | None = None,
        limited: Mapping[str, dict[str, Any]] | None = None,
        sent: list[dict[str, Any]] | None = None,
        order: list[tuple[str, int]] | None = None,
        prechecked: list[str] | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.hits = dict(cached or {})
        self.endpoints = _ENDPOINTS if endpoints is None else endpoints
        self.zdr = zdr or []
        self.blocked = dict(unavailable or {})
        self.limited = dict(limited or {})
        self.sent = sent
        self.order = order
        self.prechecked = prechecked
        self.calls = calls
        self.seen: dict[str, int] = {}

    def serve(self, request: httpx.Request) -> httpx.Response:
        """One request: a list, a lookup, a pre-check or a probe request."""
        if self.calls is not None:
            self.calls.append(request.url.path)
        listed = self._listed(request)
        if listed is not None:
            return listed
        body = json.loads(request.content)
        key = _spec_key(body)
        if self.sent is not None:
            self.sent.append(body)
        if "provibench-probe:" in request.content.decode():
            # a probe request: the first one per spec is its cold write, and a blocked tag
            # fails there the way an excluded endpoint fails a real one
            refused = self._refused(key)
            return self._probe(request, body, key) if refused is None else refused
        if self.prechecked is not None:
            self.prechecked.append(key)
        refused = self._refused(key)
        if refused is not None:
            return refused
        return httpx.Response(200, json=_answer(request, body, 0))

    def _refused(self, key: str) -> httpx.Response | None:
        """The refusal an endpoint answers every request with, or `None` when it answers."""
        if key in self.blocked:
            return httpx.Response(404, json=self.blocked[key])
        if key in self.limited:
            return httpx.Response(429, json=self.limited[key])
        return None

    def _listed(self, request: httpx.Request) -> httpx.Response | None:
        """The three GETs a run makes, or `None` when this path is a message POST.

        The endpoint list carries the health percentiles only for a request with a bearer
        token, the way OpenRouter answers it: an unkeyed fetch gets `null` for both, so a
        ranking that relies on them can only have been made from a keyed list.
        """
        path = request.url.path
        if path.endswith("/endpoints/zdr"):
            return httpx.Response(200, json={"data": self.zdr})
        if path.endswith("/endpoints"):
            keyed = bool(request.headers.get("authorization"))
            endpoints = self.endpoints if keyed else [_unkeyed(e) for e in self.endpoints]
            return httpx.Response(200, json={"data": {"endpoints": endpoints}})
        if path.endswith("/generation"):
            return httpx.Response(200, json={"data": {"id": "g1", "total_cost": 0.0001}})
        return None

    def _probe(self, request: httpx.Request, body: dict[str, Any], key: str) -> httpx.Response:
        """A probe request: the first one per spec is the cold write and reads nothing back."""
        if self.order is not None:
            self.order.append((key, _turn(body)))
        count = self.seen.get(key, 0)
        self.seen[key] = count + 1
        hit = 0 if count == 0 else self.hits.get(key, 0)
        if body.get("stream"):
            return _stream(hit, 100)
        return httpx.Response(200, json=_answer(request, body, hit))


def _unkeyed(endpoint: dict[str, Any]) -> dict[str, Any]:
    """One endpoint as an unkeyed request sees it: both 30-minute percentiles are null."""
    return {**endpoint, "latency_last_30m": None, "throughput_last_30m": None}


def _transport(**kwargs: Any) -> Callable[[httpx.Request], httpx.Response]:
    """`_Provider` as a transport handler, for `_install`."""
    return _Provider(**kwargs).serve


def _answer(request: httpx.Request, body: dict[str, Any], cached: int) -> dict[str, Any]:
    """One served response; the endpoint answers with its own model, the gateway its slug."""
    gateway = "openrouter" in request.url.host
    return {
        "id": "m1",
        "model": _MODEL if gateway else str(body.get("model")),
        "usage": {
            "input_tokens": 100 - cached,
            "cache_read_input_tokens": cached,
            "output_tokens": 1,
        },
    }


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


def _sweep(
    cli: Cli,
    bench_paths: BenchPaths,
    *args: str,
    unset: tuple[str, ...] = (),
    tty: bool = False,
) -> Any:
    trace = bench_paths.traces_dir / "t.jsonl"
    if not trace.exists():
        _write_trace(trace)
    return cli.run(
        "sweep",
        "t",
        _MODEL,
        *args,
        env={**bench_paths.env, **_ENV},
        unset=unset,
        tty_stdout=tty,
        tty_stderr=tty,
    )


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
    native = next(s for s in with_alias.document["summaries"] if s["price_source"] == "targets")
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
    sweep = meta["sweep"]
    assert sweep["model"] == _MODEL
    assert sweep["target"] == "or"
    assert sweep["included"] == ["novita"]
    assert sweep["excluded"] == ["novita/fp8"]
    # the criteria this run did not use, and the snapshot the one it did use ranked
    assert (sweep["sort"], sweep["top"], sweep["zdr"], sweep["check"]) == (
        "price",
        None,
        False,
        False,
    )
    assert sweep["dropped"] == []
    assert [ranked["tag"] for ranked in sweep["ranking"]] == ["novita"]
    assert sweep["ranking"][0]["price_input"] == pytest.approx(0.3)
    # the snapshot the estimate was priced from, one record per probed spec
    assert meta["endpoints"][f"or:{_MODEL}@novita"]["tag"] == "novita"
    native = meta["endpoints"][f"deepseek:{_NATIVE}"]
    # a native spec has no endpoint of the gateway's listing
    assert native["tag"] is None and native["quantization"] is None
    assert "source" not in native and "price_input" not in native
    native_price = meta["prices"][f"deepseek:{_NATIVE}"]
    assert native_price["source"] == "targets" and native_price["prices"]["input"] > 0
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
    assert _labels(report.document["summaries"]) == _labels(outcome.document["summaries"])


def test_report_of_a_sweep_run_reorders_by_the_price_it_recomputes(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`report` prices a sweep from its records, so the order it prints is the order that
    price gives, not the one persisted when the sweep ran under an older rule."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(cached={"novita": 90, "gmicloud": 0, _NATIVE: 100}))
    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr
    run_dir = Path(str(outcome.document["run_dir"]))
    meta = json.loads((run_dir / "run.json").read_text())
    meta["specs"] = list(reversed(meta["specs"]))
    (run_dir / "run.json").write_text(json.dumps(meta))

    report = cli.run("report", str(run_dir), env={**bench_paths.env, **_ENV})
    assert report.code == 0, report.stderr
    summaries: Any = report.document["summaries"]
    assert _labels(summaries) == _labels(outcome.document["summaries"])
    prices = [float(summary["eff_per_m_prompt"]) for summary in summaries]
    assert prices == sorted(prices)


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
    labels = [row.split("|")[0].strip() for row in spec_rows[:5]]
    assert labels[0] == f"deepseek:{_NATIVE}"  # O2: the reference row's name is never folded
    assert labels[1:] == ["@novita", "@gmicloud", "@siliconflow", "@novita/fp8"]
    # O2: the label column is now planned as wide as the reference row's own name needs
    # (never capped at what the spec table can spare the drift floor, the same trade item 8
    # made for the drift column), so every row -- header included -- can run a little past
    # `_MAX_TABLE`; this table does, by exactly the reference label's width. That is the
    # relaxed budget, not a licence for the table to grow further. The session-projection
    # footer (item 9) is prose, not a table cell, and is not held to any column budget.
    budget = 120 + len(f"deepseek:{_NATIVE}")
    for line in outcome.stdout.splitlines():
        if "|" not in line:
            continue
        without_drift = line.rsplit("|", 1)[0]
        assert len(without_drift) <= budget, line


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
    assert f"endpoints: or:{_MODEL}@<provider>" in estimate
    for tag in ("@novita", "@novita/fp8", "@siliconflow", "@gmicloud"):
        assert tag in estimate
    assert f"deepseek:{_NATIVE}" in estimate  # the native spec keeps its whole label
    assert estimate.count("openrouter-endpoint") == 4  # the gateway specs
    assert "targets" in estimate  # the native spec is priced from targets.toml


def test_sweep_separates_the_candidates_section_from_the_estimate_section(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One blank line, no more and no less, between the `candidates:` table and the
    `endpoints:` estimate table -- each is its own section (item 2's layout rule)."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(sent=[]))

    within = _sweep(cli, bench_paths, *_one_rung("--budget", "0.01"))
    assert within.code == 0, within.stderr
    lines = within.stderr.splitlines()
    candidates_at = next(i for i, line in enumerate(lines) if line.startswith("candidates:"))
    estimate_at = next(i for i, line in enumerate(lines) if line.startswith("endpoints:"))
    assert estimate_at > candidates_at
    between = lines[candidates_at + 1 : estimate_at]
    assert between[-1] == ""  # exactly one blank line closes the candidates section
    assert between[-2] != ""  # and nothing but that one blank line


def test_sweep_dry_run_sends_no_probe_request(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--dry-run` still lists the endpoints to price the estimate, but sends no probe or
    pre-check request (R4); `requires_confirmation` stays true with or without `--yes` (R3c)."""
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    calls: list[str] = []
    _install(monkeypatch, _transport(sent=sent, calls=calls))

    outcome = _sweep(
        cli,
        bench_paths,
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",
        "--dry-run",
    )
    assert outcome.code == 0, outcome.stderr
    assert sent == []  # no probe request went out
    assert all(not path.endswith("/messages") for path in calls)
    doc = outcome.document
    assert doc["partial"] is False
    assert doc["changed"] is False
    assert doc["requires_confirmation"] is True
    assert "sweep" not in doc and "run_dir" not in doc
    assert len(doc["estimate"]) >= 1  # type: ignore[arg-type]

    with_yes = _sweep(
        cli,
        bench_paths,
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",
        "--dry-run",
        "--yes",
    )
    assert with_yes.code == 0, with_yes.stderr
    assert sent == []
    assert with_yes.document["requires_confirmation"] is True


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


# --- selection: the stability floor, the sort keys, ZDR, `--top`, availability ------------


def _run_json(outcome: Any) -> dict[str, Any]:
    """A run's `run.json`, parsed."""
    path = Path(str(outcome.document["run_dir"])) / "run.json"
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _block(outcome: Any) -> dict[str, Any]:
    """The sweep block a run recorded, as the command's document carries it."""
    return SweepInfo.model_validate(_run_json(outcome)["sweep"]).to_document()


def _specs(outcome: Any) -> set[str]:
    """The labels the run directory holds, whichever order the measured price put them in."""
    return {str(ref["label"]) for ref in _run_json(outcome)["specs"]}


def _estimate_amounts(stderr: str) -> tuple[float, float]:
    """What a run's estimate printed: the total, and the share the check was priced at."""
    total = next(
        float(line.split("|")[-1].strip())
        for line in stderr.splitlines()
        if line.startswith("total") and "|" in line
    )
    share = next(line for line in stderr.splitlines() if line.startswith("pre-check: ")).rsplit(
        "$", 1
    )[1]
    return total, float(share)


def _upper_bound_amount(stderr: str) -> float:
    """The upper bound a `--top` run's estimate printed, in USD."""
    return float(
        next(line for line in stderr.splitlines() if line.startswith("upper bound if the ")).rsplit(
            "$", 1
        )[1]
    )


# Six endpoints with distinct 1-day uptimes, so `--sort uptime` has one obvious order and
# a `--top 5` still has one candidate left out. Listed in a deliberately wrong order.
_RANKED = [
    _endpoint("novita", **_health(99.9)),
    _endpoint("novita/fp8", **_health(98.5)),
    _endpoint("siliconflow", **_health(99.5)),
    _endpoint("gmicloud", **_health(99.0)),
    _endpoint("openinference", **_health(97.5)),
    _endpoint("relace/fp4", **_health(98.0)),
]
_SELECTED = ["novita", "siliconflow", "gmicloud", "novita/fp8", "relace/fp4"]


def test_sweep_top_probes_the_n_best_in_ranking_order(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--sort uptime --top 5` probes five endpoints plus the native reference, in order."""
    _write_targets(bench_paths.targets_path)
    prechecked: list[str] = []
    order: list[tuple[str, int]] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, prechecked=prechecked, order=order))

    outcome = _sweep(cli, bench_paths, *_one_rung("--sort", "uptime", "--top", "5"))
    assert outcome.code == 0, outcome.stderr
    assert _specs(outcome) == {
        *(f"or:{_MODEL}@{tag}" for tag in _SELECTED),
        f"deepseek:{_NATIVE}",
    }
    # the check ran in ranking order and stopped at the fifth kept candidate, so the sixth
    # endpoint was never checked, let alone probed; the probe kept that order too
    assert prechecked == _SELECTED
    assert [key for key, turn in order if turn == 1] == [*_SELECTED, _NATIVE]
    block = _block(outcome)
    assert (block["sort"], block["top"], block["check"]) == ("uptime", 5, True)
    assert [ranked["tag"] for ranked in block["ranking"]] == [
        "novita",
        "siliconflow",
        "gmicloud",
        "novita/fp8",
        "relace/fp4",
        "openinference",
    ]
    # the check is priced from its plan, so it counts every candidate, not the five it keeps
    assert "pre-check: up to 6 requests," in outcome.stderr


def test_sweep_price_ties_break_by_throughput_then_uptime(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default sort is price; equal prices go to the faster endpoint, then the busier one.

    Every endpoint is priced the same and listed in a deliberately wrong order, and the
    fake list serves the throughput percentiles only to a keyed request: this ordering can
    only come out right if the sweep fetched the list with the target's key.
    """
    _write_targets(bench_paths.targets_path, aliases=False)
    flat = {"prompt": "0.0000005", "completion": "0.0000005", "input_cache_read": "0"}
    endpoints = [
        _endpoint("siliconflow", pricing=flat, **_health(99.0, throughput=50.0)),
        _endpoint("novita", pricing=flat, **_health(99.9, throughput=90.0)),
        _endpoint("novita/fp8", pricing=flat, **_health(99.95)),
        _endpoint("gmicloud", pricing=flat, **_health(99.9, throughput=90.0)),
    ]
    prechecked: list[str] = []
    _install(monkeypatch, _transport(endpoints=endpoints, prechecked=prechecked))

    outcome = _sweep(cli, bench_paths, *_one_rung("--top", "4"))
    assert outcome.code == 0, outcome.stderr
    # novita and gmicloud tie on throughput and price, so their equal uptime leaves the
    # order the API listed them in; siliconflow's 50 tok/s loses to their 90, and
    # novita/fp8's unknown percentile loses to every known one
    assert prechecked == ["novita", "gmicloud", "siliconflow", "novita/fp8"]
    assert "tput p50" in outcome.stderr and "90.0" in outcome.stderr  # the keyed listing


def test_sweep_sort_latency_and_throughput_use_the_percentiles(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.0, latency=300.0, throughput=10.0)),
        _endpoint("gmicloud", **_health(99.0, latency=100.0, throughput=50.0)),
        _endpoint("siliconflow", **_health(99.0, latency=200.0, throughput=30.0)),
    ]

    for sort in ("latency", "throughput"):
        prechecked: list[str] = []
        _install(monkeypatch, _transport(endpoints=endpoints, prechecked=prechecked))
        outcome = _sweep(cli, bench_paths, *_one_rung("--sort", sort, "--top", "2"))
        assert outcome.code == 0, outcome.stderr
        # the fastest endpoint leads under either key: 100 ms and 50 tok/s belong to gmicloud
        assert prechecked == ["gmicloud", "siliconflow"]


def test_sweep_sort_by_percentile_without_the_key_names_the_flag_and_the_key(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--sort latency` ranks by data only a keyed request gets: no key, no ranking."""
    _write_targets(bench_paths.targets_path)
    calls: list[str] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, calls=calls))

    outcome = _sweep(cli, bench_paths, *_one_rung("--sort", "latency"), unset=("OR_KEY",))
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    message = str(outcome.error["message"])
    assert "--sort latency" in message and "OR_KEY" in message
    assert calls == []  # refused before the endpoint list was even fetched

    # the key is set, and the API still returns no percentile for these endpoints
    nulls = _sweep(cli, bench_paths, *_one_rung("--sort", "latency"))
    assert nulls.code == 2
    assert nulls.error["kind"] == "invalid_input"
    message = str(nulls.error["message"])
    assert "--sort latency" in message and "OR_KEY" in message and "novita" in message


def test_sweep_price_sort_without_a_key_ranks_by_uptime(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keyless list has no percentiles, and the price tie-break falls back to uptime.

    The run itself cannot go ahead — a pinned request needs the key, which is the failure
    at the end — but the listing and the ranking it is read from hold without one.
    """
    _write_targets(bench_paths.targets_path, aliases=False)
    flat = {"prompt": "0.0000005", "completion": "0.0000005", "input_cache_read": "0"}
    endpoints = [
        _endpoint("novita", pricing=flat, **_health(98.0)),
        _endpoint("siliconflow", pricing=flat, **_health(99.5)),
        _endpoint("gmicloud", pricing=flat, **_health(99.0)),
    ]
    _install(monkeypatch, _transport(endpoints=endpoints))

    outcome = _sweep(cli, bench_paths, *_one_rung(), unset=("OR_KEY",))
    assert outcome.code == 1
    assert "OR_KEY" in str(outcome.error["message"])  # the run needs it; the ranking did not
    rows = [
        line
        for line in outcome.stderr.splitlines()
        if line.startswith(("novita", "siliconflow", "gmicloud")) and "|" in line
    ]
    # equal prices, so the 1-day uptime decides: 99.5, then 99.0, then 98.0
    assert [row.split("|")[0].strip() for row in rows] == [
        "siliconflow",
        "gmicloud",
        "novita",
    ]
    # and both percentile columns are the nulls the unkeyed API sent
    assert all(row.rstrip().endswith("-") for row in rows)


def test_sweep_stability_floor_drops_endpoints_and_include_brings_one_back(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor is always on; `--include` is the one way back in for a dropped tag."""
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("relace/fp4", **_health(99.9, status=-2)),
        _endpoint("siliconflow", **_health(77.9)),
        # the live case: 96.96 rounds to the floor at one decimal, so the reason spells it out
        _endpoint("parasail/fp8", **_health(96.96)),
        _endpoint("gmicloud", **_health(99.0)),
    ]
    _install(monkeypatch, _transport(endpoints=endpoints))

    outcome = _sweep(cli, bench_paths, *_one_rung())
    assert outcome.code == 0, outcome.stderr
    assert _specs(outcome) == {f"or:{_MODEL}@novita", f"or:{_MODEL}@gmicloud"}
    assert "dropped: relace/fp4 (status -2), siliconflow (uptime 1d 77.90 %)" in outcome.stderr
    assert "parasail/fp8 (uptime 1d 96.96 %)" in outcome.stderr
    assert [(drop["tag"], drop["reason"]) for drop in _block(outcome)["dropped"]] == [
        ("relace/fp4", "status -2"),
        ("siliconflow", "uptime 1d 77.90 %"),
        ("parasail/fp8", "uptime 1d 96.96 %"),
    ]

    included = _sweep(cli, bench_paths, *_one_rung("--include", "relace"))
    assert included.code == 0, included.stderr
    assert _specs(included) == {f"or:{_MODEL}@relace/fp4"}
    assert "dropped:" not in included.stderr


def test_sweep_min_uptime_moves_the_floor(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--min-uptime` replaces the default 97 % floor everywhere the floor is used."""
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("relace/fp4", **_health(99.9, status=-2)),
        _endpoint("siliconflow", **_health(77.9)),
        _endpoint("parasail/fp8", **_health(96.96)),
        _endpoint("gmicloud", **_health(99.0)),
    ]
    _install(monkeypatch, _transport(endpoints=endpoints))

    # the flag's default is the floor the run records, so a report can name it
    default = _sweep(cli, bench_paths, *_one_rung())
    assert default.code == 0, default.stderr
    assert _block(default)["uptime_floor"] == 97.0

    # a lower floor keeps parasail/fp8 (96.96) that the default 97 % drops, and the listing
    # names the floor the drops were held to
    lower = _sweep(cli, bench_paths, *_one_rung("--min-uptime", "95"))
    assert lower.code == 0, lower.stderr
    assert "sort=price, zdr=off, min-uptime=95" in lower.stderr
    assert _specs(lower) == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@parasail/fp8",
        f"or:{_MODEL}@gmicloud",
    }
    assert [drop["tag"] for drop in _block(lower)["dropped"]] == ["relace/fp4", "siliconflow"]
    assert _block(lower)["uptime_floor"] == 95.0

    # a higher floor drops gmicloud (99.0) that the default keeps
    higher = _sweep(cli, bench_paths, *_one_rung("--min-uptime", "99.5"))
    assert higher.code == 0, higher.stderr
    assert _specs(higher) == {f"or:{_MODEL}@novita"}
    assert "gmicloud (uptime 1d 99.00 %)" in higher.stderr
    assert _block(higher)["uptime_floor"] == 99.5

    # the selection sentence names the floor this run actually used
    run_dir = str(higher.document["run_dir"])
    printed = cli.run("report", run_dir, tty_stdout=True, env={**bench_paths.env, **_ENV})
    assert printed.code == 0, printed.stderr
    assert "the 99.5 % floor" in printed.stdout

    # --include still overrides the floor once the floor has been raised
    kept = _sweep(cli, bench_paths, *_one_rung("--min-uptime", "99.5", "--include", "gmicloud"))
    assert kept.code == 0, kept.stderr
    assert _specs(kept) == {f"or:{_MODEL}@gmicloud"}

    # a floor of 0 drops nothing for uptime; the status floor still applies
    zeroed = _sweep(cli, bench_paths, *_one_rung("--min-uptime", "0"))
    assert zeroed.code == 0, zeroed.stderr
    assert [(drop["tag"], drop["reason"]) for drop in _block(zeroed)["dropped"]] == [
        ("relace/fp4", "status -2"),
    ]
    assert _block(zeroed)["uptime_floor"] == 0.0


def test_sweep_min_uptime_out_of_range_is_a_usage_error(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(sent=sent))

    for value in ("101", "-1"):
        outcome = _sweep(cli, bench_paths, *_one_rung("--min-uptime", value))
        assert outcome.code == 2, value
        assert outcome.error["kind"] == "invalid_input", value
    assert sent == []


def test_sweep_zdr_keeps_only_the_endpoints_on_the_list(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    zdr = [
        {"model_id": _MODEL, "tag": "novita"},
        {"model_id": _MODEL, "tag": "gmicloud"},
        {"model_id": "other/model", "tag": "siliconflow"},  # another model's endpoint
    ]
    _install(monkeypatch, _transport(endpoints=_RANKED, zdr=zdr))

    outcome = _sweep(cli, bench_paths, *_one_rung("--zdr"))
    assert outcome.code == 0, outcome.stderr
    assert _specs(outcome) == {f"or:{_MODEL}@novita", f"or:{_MODEL}@gmicloud"}
    assert "zdr=on" in outcome.stderr
    assert "dropped: novita/fp8 (not ZDR), siliconflow (not ZDR), openinference (not ZDR)" in (
        outcome.stderr
    )
    assert "relace/fp4 (not ZDR)" in outcome.stderr
    assert _block(outcome)["zdr"] is True


def test_sweep_quantization_keeps_only_the_named_quantizations(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", quantization="fp8"),
        _endpoint("relace/fp4", quantization="fp4"),
        _endpoint("siliconflow", quantization=None),
    ]
    _install(monkeypatch, _transport(endpoints=endpoints))

    outcome = _sweep(cli, bench_paths, *_one_rung("--quantization", "fp8"))
    assert outcome.code == 0, outcome.stderr
    assert _specs(outcome) == {f"or:{_MODEL}@novita"}
    assert [(drop["tag"], drop["reason"]) for drop in _block(outcome)["dropped"]] == [
        ("relace/fp4", "quantization fp4"),
        ("siliconflow", "quantization unknown"),
    ]
    assert _block(outcome)["quantization"] == ["fp8"]
    assert "quant=fp8" in outcome.stderr

    # comma-separated in one occurrence keeps the unlabelled endpoint too
    both = _sweep(cli, bench_paths, *_one_rung("--quantization", "fp8,unknown"))
    assert both.code == 0, both.stderr
    assert _specs(both) == {f"or:{_MODEL}@novita", f"or:{_MODEL}@siliconflow"}

    # --include overrides the stability floor, not the quantization filter: one axis at a time
    included = _sweep(
        cli,
        bench_paths,
        *_one_rung("--quantization", "fp8", "--include", "relace", "--include", "novita"),
    )
    assert included.code == 0, included.stderr
    assert f"or:{_MODEL}@relace/fp4" not in _specs(included)
    assert f"or:{_MODEL}@novita" in _specs(included)


def test_sweep_pin_probes_the_top_n_plus_the_pin(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--pin TAG --top 2` probes the two best plus the pin, outside the ranking's cut.

    Regression: with more ranked candidates ahead of the pin than `--top` needs, a `keep`
    cut applied to a single list holding both the ranking and the pin stops before it ever
    reaches the pin, because two of the three ranked candidates here already satisfy it.
    """
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("novita/fp8", **_health(99.5)),
        _endpoint("siliconflow", **_health(99.0)),
        _endpoint("gmicloud", **_health(80.0)),
    ]
    prechecked: list[str] = []
    _install(monkeypatch, _transport(endpoints=endpoints, prechecked=prechecked))

    outcome = _sweep(
        cli, bench_paths, *_one_rung("--sort", "uptime", "--top", "2", "--pin", "gmicloud")
    )
    assert outcome.code == 0, outcome.stderr
    # the ranked check stops after two survive (novita, novita/fp8); siliconflow is never
    # visited, and the pin is checked afterward regardless
    assert prechecked == ["novita", "novita/fp8", "gmicloud"]
    assert _specs(outcome) == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@novita/fp8",
        f"or:{_MODEL}@gmicloud",
    }
    assert _block(outcome)["dropped"] == []
    assert _block(outcome)["pinned"] == ["gmicloud"]
    assert "pin=gmicloud" in outcome.stderr


def test_sweep_pin_that_404s_is_dropped_without_promoting_a_ranked_endpoint(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin the account can't reach is a checked drop; it never frees a --top slot, since
    it was never competing with the ranked candidates for one."""
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("siliconflow", **_health(99.5)),
        _endpoint("openinference", **_health(99.0)),
        _endpoint("gmicloud", **_health(80.0)),
    ]
    prechecked: list[str] = []
    _install(
        monkeypatch,
        _transport(
            endpoints=endpoints,
            prechecked=prechecked,
            unavailable={"gmicloud": _GUARDRAIL_404},
        ),
    )

    outcome = _sweep(
        cli, bench_paths, *_one_rung("--sort", "uptime", "--top", "2", "--pin", "gmicloud")
    )
    assert outcome.code == 0, outcome.stderr
    # novita and siliconflow already satisfy --top 2; openinference is never checked or
    # probed even though the pin later fails
    assert prechecked == ["novita", "siliconflow", "gmicloud"]
    assert _specs(outcome) == {f"or:{_MODEL}@novita", f"or:{_MODEL}@siliconflow"}
    dropped = _block(outcome)["dropped"]
    assert [(drop["tag"], drop["checked"]) for drop in dropped] == [("gmicloud", True)]
    assert _GUARDRAIL_REASON in outcome.stderr


def test_sweep_pin_is_additive_over_include(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--include` narrows the listing to matching tags; a `--pin` outside that prefix is
    added on top, not an error -- the caller named it directly rather than by prefix."""
    _write_targets(bench_paths.targets_path, aliases=False)
    _install(monkeypatch, _transport(endpoints=_ENDPOINTS))

    outcome = _sweep(cli, bench_paths, *_one_rung("--include", "novita", "--pin", "gmicloud"))
    assert outcome.code == 0, outcome.stderr
    assert _specs(outcome) == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@novita/fp8",
        f"or:{_MODEL}@gmicloud",
    }


def test_sweep_pin_below_the_uptime_floor_is_kept(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("gmicloud", **_health(50.0)),  # well below the 97 % default floor
    ]
    _install(monkeypatch, _transport(endpoints=endpoints))

    outcome = _sweep(cli, bench_paths, *_one_rung("--pin", "gmicloud"))
    assert outcome.code == 0, outcome.stderr
    assert _specs(outcome) == {f"or:{_MODEL}@novita", f"or:{_MODEL}@gmicloud"}
    assert _block(outcome)["dropped"] == []


def test_sweep_pin_of_an_unknown_tag_is_a_usage_error(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    _install(monkeypatch, _transport(endpoints=_ENDPOINTS))

    outcome = _sweep(cli, bench_paths, *_one_rung("--pin", "ghost"))
    assert outcome.code == 2, outcome.stderr
    assert outcome.error["kind"] == "invalid_input"


def test_sweep_pin_and_exclude_of_the_same_tag_is_a_usage_error(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_targets(bench_paths.targets_path, aliases=False)
    _install(monkeypatch, _transport(endpoints=_ENDPOINTS))

    outcome = _sweep(cli, bench_paths, *_one_rung("--pin", "novita", "--exclude", "novita"))
    assert outcome.code == 2, outcome.stderr
    assert outcome.error["kind"] == "invalid_input"
    # the real reason, not the "matches no endpoint" the exclude-then-pin order would give
    message = str(outcome.error["message"])
    assert "--pin 'novita'" in message
    assert "--exclude 'novita'" in message


def test_sweep_dry_run_estimate_includes_the_pinned_endpoint(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `--dry-run` estimate prices the two best ranked candidates plus the pin -- not a
    third ranked endpoint the plain `--top` cut would otherwise have priced -- and the
    candidates table names both. `total_usd` is summed from exactly those rows."""
    _write_targets(bench_paths.targets_path, aliases=False)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("siliconflow", **_health(99.5)),
        _endpoint("openinference", **_health(99.0)),
        _endpoint("gmicloud", **_health(80.0)),
    ]
    _install(monkeypatch, _transport(endpoints=endpoints))

    outcome = _sweep(
        cli,
        bench_paths,
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",
        "--sort",
        "uptime",
        "--top",
        "2",
        "--pin",
        "gmicloud",
        "--dry-run",
    )
    assert outcome.code == 0, outcome.stderr
    rows = [d for d in map(as_document, as_list(outcome.document["estimate"]) or []) if d]
    assert {str(row["label"]) for row in rows} == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@siliconflow",
        f"or:{_MODEL}@gmicloud",
    }
    assert "quant" in outcome.stderr  # the candidates table's column header
    assert "pin=gmicloud" in outcome.stderr

    # the availability check's own cost still prices its worst case (every ranked candidate
    # it might have to try, plus the pin); only the estimate rows above are narrowed to the
    # three the run actually means to probe
    pre_check = as_document(outcome.document["pre_check"])
    assert pre_check is not None

    worst_cases = [row.get("worst_case_usd") for row in rows]
    assert all(isinstance(amount, int | float) for amount in worst_cases)
    total = outcome.document["total_usd"]
    assert isinstance(total, int | float)
    assert total == pytest.approx(
        sum(float(str(amount)) for amount in worst_cases) + float(str(pre_check["usd"]))
    )


def test_sweep_schema_lists_quantization_and_pinned(cli: Cli) -> None:
    detail = cli.run("schema", "sweep")
    assert detail.code == 0, detail.stderr
    printed = json.dumps(detail.document)
    assert "quantization" in printed
    assert "pinned" in printed
    assert "reported_average" in printed
    for field in (
        "or_avg_share_pct",
        "or_avg_pooled",
        "or_avg_eff_per_m_prompt",
        "or_avg_session_prompt_usd",
        "vs_or_avg_pct",
    ):
        assert field in printed
    # `model` and `permaslug` also print elsewhere in the schema (the sweep's own model, the
    # candidate list), so `reported_average`'s own properties are checked by name, not by a
    # bare substring of the whole document.
    output = as_document(detail.document["output"])
    assert output is not None
    properties = as_document(output["properties"])
    assert properties is not None
    reported_average_schema = as_document(properties["reported_average"])
    assert reported_average_schema is not None
    schema_properties = as_document(reported_average_schema["properties"])
    assert schema_properties is not None
    assert {"day", "model", "permaslug", "fetched_at", "shares"} <= set(schema_properties)


def test_sweep_availability_check_removes_a_candidate_without_a_summary_row(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 404 from the account's settings is a note, not a slot: `--top 2` still probes two."""
    _write_targets(bench_paths.targets_path)
    endpoints = [
        _endpoint("novita", **_health(99.9)),
        _endpoint("gmicloud", **_health(99.5)),
        _endpoint("siliconflow", **_health(99.0)),
        _endpoint("novita/fp8", **_health(98.5)),
    ]
    prechecked: list[str] = []
    _install(
        monkeypatch,
        _transport(
            endpoints=endpoints,
            prechecked=prechecked,
            unavailable={"gmicloud": _GUARDRAIL_404},
        ),
    )

    outcome = _sweep(cli, bench_paths, *_one_rung("--sort", "uptime", "--top", "2"))
    assert outcome.code == 0, outcome.stderr
    assert prechecked == ["novita", "gmicloud", "siliconflow"]
    assert _specs(outcome) == {
        f"or:{_MODEL}@novita",
        f"or:{_MODEL}@siliconflow",
        f"deepseek:{_NATIVE}",
    }
    assert _GUARDRAIL_REASON in outcome.stderr
    # the estimate prices the check from its plan, so it counts all four candidates, the
    # one the check removed included: the run stops early only once `--top` have survived
    assert "pre-check: up to 4 requests," in outcome.stderr
    assert _block(outcome)["dropped"] == [
        {
            "tag": "gmicloud",
            "endpoint": f"or:{_MODEL}@gmicloud",
            "reason": _GUARDRAIL_REASON,
            "checked": True,
        }
    ]


def test_sweep_precheck_names_a_wrapped_429_like_the_probe_does(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check's drop reason reads the record's error the way the probe's skip note does.

    A 429 comes back inside the gateway's own envelope, so the body's own `type` says
    `error`; the failure is named by the object it wraps, which is what the record keeps.
    """
    _write_targets(bench_paths.targets_path, aliases=False)
    prechecked: list[str] = []

    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)  # the 429 backoff, without the waiting
    _install(
        monkeypatch,
        _transport(
            endpoints=_RANKED,
            prechecked=prechecked,
            limited={"gmicloud": _RATE_LIMIT_429},
        ),
    )

    outcome = _sweep(cli, bench_paths, *_one_rung("--sort", "uptime", "--top", "3"))
    assert outcome.code == 0, outcome.stderr
    # the rate-limited candidate is out and the next ranked one takes its slot; the three
    # requests to it are the attempt and its two retries
    assert list(dict.fromkeys(prechecked)) == ["novita", "siliconflow", "gmicloud", "novita/fp8"]
    assert prechecked.count("gmicloud") == 3
    assert _RATE_LIMIT_REASON in outcome.stderr
    dropped = _block(outcome)["dropped"]
    assert [drop["tag"] for drop in dropped] == ["gmicloud"]
    assert dropped[0]["reason"] == _RATE_LIMIT_REASON and dropped[0]["checked"] is True


def test_sweep_check_without_top_checks_every_candidate(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--check` alone is the pre-check without a cut: everything that answers is probed."""
    _write_targets(bench_paths.targets_path, aliases=False)
    prechecked: list[str] = []
    _install(
        monkeypatch,
        _transport(
            endpoints=_RANKED,
            prechecked=prechecked,
            unavailable={"novita/fp8": _GUARDRAIL_404},
        ),
    )

    outcome = _sweep(cli, bench_paths, *_one_rung("--check"))
    assert outcome.code == 0, outcome.stderr
    # every candidate is checked, and the default sort still decides the order: cheapest first
    assert prechecked == [
        "openinference",
        "siliconflow",
        "novita",
        "gmicloud",
        "novita/fp8",
        "relace/fp4",
    ]
    assert _specs(outcome) == {
        *(
            f"or:{_MODEL}@{tag}"
            for tag in ("openinference", "siliconflow", "novita", "gmicloud", "relace/fp4")
        ),
    }
    assert _block(outcome)["check"] is True


def test_sweep_availability_check_counts_in_the_estimate_and_the_budget(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate prices the check from its plan, and `--budget` reads that one total."""
    _write_targets(bench_paths.targets_path)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, sent=sent))

    refused = _sweep(cli, bench_paths, *_one_rung("--top", "4", "--budget", "0.0001"))
    assert refused.code == 2
    assert refused.error["kind"] == "invalid_input"
    assert "exceeds --budget" in str(refused.error["message"])
    assert sent == []  # nothing is sent before the budget has been checked

    within = _sweep(cli, bench_paths, *_one_rung("--top", "4", "--budget", "1"))
    assert within.code == 0, within.stderr
    # turn 1 records 100 prompt tokens, so the plan's six candidate checks are 600 tokens
    assert "pre-check: up to 6 requests, 600 tokens," in within.stderr


def test_top_sweep_pre_check_spend_lands_in_the_run_total_exactly_once(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check's own spend is folded into `spend_usd` once: not missing, not doubled.

    O5: the availability check's requests are priced by `report_spend` alongside every
    probed spec's own records, so the run's published total has to equal exactly the sum of
    the two -- neither the check's share left out nor counted twice because both paths touch
    the same `precheck.jsonl`.
    """
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(endpoints=_RANKED))

    outcome = _sweep(cli, bench_paths, *_one_rung("--top", "4", "--budget", "1"))
    assert outcome.code == 0, outcome.stderr

    precheck = outcome.document["precheck"]
    assert precheck is not None
    precheck_spend = precheck["spend_usd"]
    assert precheck_spend is not None and precheck_spend > 0

    summaries_spend = sum(
        summary["spend_usd"]
        for summary in outcome.document["summaries"]
        if summary["spend_usd"] is not None
    )
    assert summaries_spend > 0  # the probed specs themselves billed something too

    assert outcome.document["spend_usd"] == pytest.approx(summaries_spend + precheck_spend)


def test_sweep_top_budget_must_cover_the_upper_bound(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--budget` has to cover the priciest candidates the check can promote, not the N best.

    The check removes candidates after the estimate was agreed to, so the run can probe
    endpoints the estimate never priced — here `relace/fp4` and `novita/fp8` instead of the
    cheaper `openinference` and `siliconflow` the price sort put first.
    """
    _write_targets(bench_paths.targets_path, aliases=False)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, sent=sent))

    priced = _sweep(cli, bench_paths, *_one_rung("--top", "2", "--budget", "1"))
    assert priced.code == 0, priced.stderr
    total, _ = _estimate_amounts(priced.stderr)
    upper = _upper_bound_amount(priced.stderr)
    assert total < upper  # the two best-ranked candidates are not the two priciest
    sent.clear()

    between = _sweep(
        cli, bench_paths, *_one_rung("--top", "2", "--budget", f"{(total + upper) / 2:.6f}")
    )
    assert between.code == 2
    assert between.error["kind"] == "invalid_input"
    message = str(between.error["message"])
    assert "upper bound" in message and f"${upper:.4f}" in message
    assert sent == []  # refused before anything went out


def test_sweep_prints_no_upper_bound_when_it_equals_the_total(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that cuts nothing has one number: every candidate is already priced."""
    _write_targets(bench_paths.targets_path, aliases=False)
    _install(monkeypatch, _transport(endpoints=_RANKED))

    # without `--top` there is no cut at all, and `--top` on every candidate is the same run
    for flags in (("--check",), ("--top", "6")):
        outcome = _sweep(cli, bench_paths, *_one_rung(*flags, "--budget", "1"))
        assert outcome.code == 0, outcome.stderr
        assert "upper bound" not in outcome.stderr


def test_sweep_nothing_is_sent_when_the_confirmation_is_declined(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check is planned before the confirmation and sent after it, never before it."""
    _write_targets(bench_paths.targets_path)
    _write_trace(bench_paths.traces_dir / "t.jsonl")
    sent: list[dict[str, Any]] = []
    prechecked: list[str] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, sent=sent, prechecked=prechecked))

    outcome = cli.run(
        "sweep",
        "t",
        _MODEL,
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",  # no --yes: the prompt is the confirmation
        "--top",
        "2",
        stdin="n\n",
        tty_stdin=True,
        tty_stderr=True,
        env={**bench_paths.env, **_ENV},
    )
    assert outcome.code == 2
    assert outcome.error["kind"] == "confirmation_required"
    assert prechecked == [] and sent == []  # nothing at all went out
    # the question names the check the estimate priced, and it is asked before any of it;
    # it also carries the upper bound, which is what a `--top` run can really be billed
    assert "the availability check runs first, up to 6 candidate(s)" in outcome.stderr
    assert "if the check promotes the 2 priciest candidates" in outcome.stderr
    assert "pre-check: up to 6 requests," in outcome.stderr


def test_sweep_budget_refusal_names_the_check_that_tipped_it(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget the specs alone fit says the availability check is what it cannot cover.

    The check is what `--check` adds on its own; under `--top` a refusal names the upper
    bound instead, because the check can promote candidates the estimate did not price.
    """
    _write_targets(bench_paths.targets_path, aliases=False)
    sent: list[dict[str, Any]] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, sent=sent))

    priced = _sweep(cli, bench_paths, *_one_rung("--check", "--budget", "1"))
    assert priced.code == 0, priced.stderr
    total, share = _estimate_amounts(priced.stderr)
    specs_only = total - share
    assert 0 < share < total  # the plan is a real share of what the run costs
    sent.clear()

    tipped = _sweep(
        cli, bench_paths, *_one_rung("--check", "--budget", f"{specs_only + share / 2:.6f}")
    )
    assert tipped.code == 2
    assert "of it the availability check" in str(tipped.error["message"])

    plain = _sweep(cli, bench_paths, *_one_rung("--check", "--budget", f"{specs_only / 2:.6f}"))
    assert plain.code == 2
    assert "of it the availability check" not in str(plain.error["message"])
    assert sent == []  # refused before anything went out, both times


def test_sweep_failure_output_keeps_the_selection_lines(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that lost a request prints its tables *and* how its endpoints were chosen.

    The reader of a partial run is the one who most needs to know which candidates the
    check removed before the requests that failed, and a failing run never reaches the
    renderer that would otherwise have printed it.
    """
    _write_targets(bench_paths.targets_path)
    _install(
        monkeypatch,
        _transport(
            endpoints=_RANKED,
            # the candidate the check removes, and the reference whose request then fails
            unavailable={"novita/fp8": _GUARDRAIL_404, _NATIVE: _GUARDRAIL_404},
        ),
    )

    outcome = _sweep(cli, bench_paths, *_one_rung("--check"), tty=True)
    assert outcome.code == 1
    assert "hit %" in outcome.stderr  # the tables the text failure path prints
    assert "Of the OpenRouter providers, the run took every candidate by price." in outcome.stderr
    assert f"novita/fp8 was skipped because {_GUARDRAIL_REASON}." in outcome.stderr


def test_sweep_partial_failure_reaches_stdout_as_json(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that lost a request is O5a's partial result: the document, `partial: true`
    included, is still written to stdout, and the `operation_failed` error on stderr names
    only the run, not the document again (fix)."""
    _write_targets(bench_paths.targets_path)
    _install(
        monkeypatch,
        _transport(
            endpoints=_RANKED,
            unavailable={"novita/fp8": _GUARDRAIL_404, _NATIVE: _GUARDRAIL_404},
        ),
    )

    outcome = _sweep(cli, bench_paths, *_one_rung("--check"))
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"
    context = as_document(outcome.error.get("context"))
    assert context is not None
    assert set(context) == {"run_dir", "run_hex"}
    doc = outcome.document
    assert doc["partial"] is True
    assert doc["sweep"] is not None


def test_report_shows_the_selection_of_a_sweep_offline(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`report` prints the criteria, the drops and the ranking from the run directory."""
    _write_targets(bench_paths.targets_path)
    _install(monkeypatch, _transport(endpoints=_RANKED, unavailable={"gmicloud": _GUARDRAIL_404}))
    outcome = _sweep(cli, bench_paths, *_one_rung("--sort", "uptime", "--top", "3"))
    assert outcome.code == 0, outcome.stderr

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"report went to the network: {request.url}")

    _install(monkeypatch, refuse)
    run_dir = str(outcome.document["run_dir"])
    documented = cli.run("report", run_dir, env={**bench_paths.env, **_ENV})
    assert documented.code == 0, documented.stderr
    assert documented.document["sweep"] == outcome.document["sweep"]

    printed = cli.run("report", run_dir, tty_stdout=True, env={**bench_paths.env, **_ENV})
    assert printed.code == 0, printed.stderr
    assert "Of the OpenRouter providers, the run took the 3 best by uptime." in printed.stdout
    assert f"gmicloud was skipped because {_GUARDRAIL_REASON}." in printed.stdout

    # the file shared around carries the same lines, from the same document
    html_path = bench_paths.runs_dir.parent / "report.html"
    written = cli.run(
        "report",
        run_dir,
        "--format",
        "html",
        "--output-file",
        str(html_path),
        env={**bench_paths.env, **_ENV},
    )
    assert written.code == 0, written.stderr
    html = html_path.read_text(encoding="utf-8")
    assert "Of the OpenRouter providers, the run took the 3 best by uptime." in html
    assert f"gmicloud was skipped because {_GUARDRAIL_REASON}." in html


def test_report_of_a_probe_by_hand_has_no_selection(
    cli: Cli, bench_paths: BenchPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe whose specs were named by hand chose nothing, so nothing is reported."""
    _write_targets(bench_paths.targets_path)
    _write_trace(bench_paths.traces_dir / "t.jsonl")
    prechecked: list[str] = []
    _install(monkeypatch, _transport(endpoints=_RANKED, prechecked=prechecked))
    probe = cli.run(
        "probe",
        "t",
        f"or:{_MODEL}@novita",
        "--rungs",
        "1",
        "--repeats",
        "1",
        "--gap",
        "0",
        "--no-throughput",
        "--yes",
        env={**bench_paths.env, **_ENV},
    )
    assert probe.code == 0, probe.stderr
    assert "sweep" not in probe.document
    assert prechecked == []  # no availability check without --top or --check

    run_dir = str(probe.document["run_dir"])
    documented = cli.run("report", run_dir, env={**bench_paths.env, **_ENV})
    assert documented.code == 0, documented.stderr
    assert documented.document["sweep"] is None

    printed = cli.run("report", run_dir, tty_stdout=True, env={**bench_paths.env, **_ENV})
    assert printed.code == 0, printed.stderr
    assert "Of the OpenRouter providers" not in printed.stdout
