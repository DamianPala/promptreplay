"""`endpoints`: fetching, sorting, and shaping OpenRouter endpoint listings.

`bench.openrouter.parse_endpoints` itself is unit-tested elsewhere; these tests mock only
the network boundary (`httpx.AsyncClient`) and exercise the CLI's sort options, error
mapping, and the `status` int/str normalisation.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from promptreplay.core.documents import Document, as_document, as_list
from tests.conftest import Cli

_RealAsyncClient = httpx.AsyncClient

_PAYLOAD = {
    "data": {
        "endpoints": [
            {
                "provider_name": "Cheap",
                "tag": "cheap",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                "uptime_last_30m": 90.0,
                "latency_last_30m": 500.0,
                "throughput_last_30m": 10.0,
                "status": 200,
            },
            {
                "provider_name": "Pricey",
                "tag": "pricey",
                "pricing": {"prompt": "0.0000005", "completion": "0.0000009"},
                "uptime_last_30m": 99.0,
                "latency_last_30m": 100.0,
                "throughput_last_30m": 50.0,
                "status": 200,
            },
        ]
    }
}


def _client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.AsyncClient]:
    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)  # type: ignore[arg-type]

    return factory


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_PAYLOAD)


def _endpoints(document: Document) -> list[Document]:
    return [d for d in map(as_document, as_list(document["endpoints"]) or []) if d]


def _tags(document: Document) -> list[str]:
    return [str(entry["tag"]) for entry in _endpoints(document)]


def test_endpoints_default_sort_is_price_ascending(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))
    outcome = cli.run("endpoints", "some/model")
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert doc["model"] == "some/model"
    assert _tags(doc) == ["cheap", "pricey"]
    first = _endpoints(doc)[0]
    assert first["status"] == "200"
    prices = as_document(first["prices"]) or {}
    assert prices["input"] == pytest.approx(0.1)


def test_endpoints_sort_by_uptime_throughput_and_latency(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))
    by_uptime = cli.run("endpoints", "some/model", "--sort", "uptime")
    assert _tags(by_uptime.document) == ["pricey", "cheap"]
    by_throughput = cli.run("endpoints", "some/model", "--sort", "throughput")
    assert _tags(by_throughput.document) == ["pricey", "cheap"]
    by_latency = cli.run("endpoints", "some/model", "--sort", "latency")
    assert _tags(by_latency.document) == ["pricey", "cheap"]


def test_endpoints_upstream_error_is_operation_failed(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(failing))
    outcome = cli.run("endpoints", "some/model")
    assert outcome.code == 1
    assert outcome.error["kind"] == "operation_failed"


def test_endpoints_not_found_for_an_unknown_slug(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad slug is a clean not_found, not the raw httpx 404 text (change request 16c)."""

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html>404 Not Found ... developer.mozilla.org ...</html>")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(missing))
    outcome = cli.run("endpoints", "deepseek/no-such-model")
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"
    assert (
        outcome.error["message"] == "OpenRouter lists no model with slug 'deepseek/no-such-model'"
    )
    assert "mozilla" not in outcome.stderr


def test_endpoints_not_found_hints_same_author_slugs(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 404 gets a hint naming the same author's other slugs, ranked by how much of the
    requested slug each one shares as a prefix, longest first, capped at 6 (agent-test
    finding 6a)."""
    models_payload = {
        "data": [
            {"id": "deepseek/deepseek-v4.1-flash"},
            {"id": "deepseek/deepseek-v4.1-terminus"},
            {"id": "deepseek/deepseek-v4.1-base"},
            {"id": "deepseek/deepseek-v4.1-mini"},
            {"id": "deepseek/deepseek-v4.1-nano"},
            {"id": "deepseek/deepseek-v4.1"},
            {"id": "deepseek/deepseek-v3"},
            {"id": "openai/gpt-5"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            assert "authorization" not in request.headers
            return httpx.Response(200, json=models_payload)
        return httpx.Response(404, text="not found")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    outcome = cli.run("endpoints", "deepseek/deepseek-v4.1-flash-typo")
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"
    assert outcome.error["hint"] == (
        "Slugs by the same author: deepseek/deepseek-v4.1-flash, "
        "deepseek/deepseek-v4.1-terminus, deepseek/deepseek-v4.1-base, "
        "deepseek/deepseek-v4.1-mini, deepseek/deepseek-v4.1-nano, "
        "deepseek/deepseek-v4.1"
    )


def test_endpoints_not_found_has_no_hint_without_an_author_in_the_slug(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slug with no `/` has no author to look up, so the hint is skipped without a request
    to the models list."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/api/v1/models"
        return httpx.Response(404, text="not found")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    outcome = cli.run("endpoints", "no-such-slug")
    assert outcome.code == 1
    assert "hint" not in outcome.error


def test_endpoints_not_found_has_no_hint_when_the_models_list_also_fails(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The models-list lookup is best-effort: any failure there falls back to the plain
    not_found instead of a second error on top of it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    outcome = cli.run("endpoints", "deepseek/deepseek-v4.1-flash-typo")
    assert outcome.code == 1
    assert "hint" not in outcome.error


def test_endpoints_not_found_has_no_hint_for_a_models_payload_that_is_not_an_object(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 whose body is a bare list, not `{"data": [...]}`, is still a failed lookup: the
    404 comes back without a hint rather than as an unhandled error on top of it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json=["deepseek/deepseek-v4.1"])
        return httpx.Response(404, text="not found")

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(handler))
    outcome = cli.run("endpoints", "deepseek/deepseek-v4.1-flash-typo")
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"
    assert "hint" not in outcome.error


def test_endpoints_sends_the_default_gateway_key_when_set(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a key, OpenRouter never returns latency/throughput percentiles (change request
    16 item 8): `endpoints` reads the same default key `sweep`'s unpinned listing would."""
    seen: list[str | None] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_PAYLOAD)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(capture))
    outcome = cli.run("endpoints", "some/model", env={"OPENROUTER_API_KEY": "secret-key"})
    assert outcome.code == 0, outcome.stderr
    assert seen == ["Bearer secret-key"]


def test_endpoints_human_table(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))
    outcome = cli.run("endpoints", "some/model", tty=True)
    assert outcome.code == 0
    assert "cheap" in outcome.stdout and "pricey" in outcome.stdout


def test_endpoints_has_no_leading_or_trailing_blank_line(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))
    outcome = cli.run("endpoints", "some/model", tty=True)
    assert outcome.code == 0
    lines = outcome.stdout.splitlines()
    assert lines[0] != "" and lines[-1] != ""
    assert "" not in lines
    assert [cell.strip() for cell in lines[0].split(" | ")][:2] == ["tag", "quant"]


def test_endpoints_table_drops_provider_and_context_and_fits_120_columns(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text table used to render 164 columns wide on real (long) tags and provider
    names and wrap on a 120-column terminal; `provider` and `context` are dropped there
    (the tag already names the provider) and the remaining headers are shortened to the
    ones `sweep`'s candidate table uses."""
    payload = {
        "data": {
            "endpoints": [
                {
                    "provider_name": "DeepInfra Serverless Quantized",
                    "tag": "novita",
                    "pricing": {"prompt": "0.0000003", "completion": "0.0000006"},
                    "context_length": 128_000,
                    "uptime_last_30m": 99.87,
                    "uptime_last_1d": 99.5,
                    "latency_last_30m": 1234.5,
                    "throughput_last_30m": 123.4,
                    "status": 200,
                }
            ]
        }
    }

    def big(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(big))
    outcome = cli.run("endpoints", "deepseek/deepseek-v4.1-flash", tty=True)
    assert outcome.code == 0, outcome.stderr
    lines = outcome.stdout.splitlines()
    assert "provider" not in lines[0].split(" | ") and "context" not in lines[0].split(" | ")
    header = [cell.strip() for cell in lines[0].split(" | ")]
    assert header == [
        "tag",
        "quant",
        "in",
        "cache read",
        "cache write",
        "out",
        "uptime 30m",
        "uptime 1d",
        "lat p50 ms",
        "tput p50",
        "impl. cache",
    ]
    assert all(len(line) <= 120 for line in lines)
