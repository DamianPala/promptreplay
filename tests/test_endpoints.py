"""`endpoints`: fetching, sorting, and shaping OpenRouter endpoint listings.

`bench.openrouter.parse_endpoints` itself is unit-tested elsewhere; these tests mock only
the network boundary (`httpx.AsyncClient`) and exercise the CLI's sort options, error
mapping, and the `status` int/str normalisation.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from provibench.core.documents import Document, as_document, as_list
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


def test_endpoints_human_table(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(_ok))
    # rich sizes an unfiled (non-fileno) console from the real `os.environ["COLUMNS"]`,
    # not the injected process env; without widening it, this 13-column table would
    # truncate every cell to an ellipsis and leave nothing recognisable to assert on.
    monkeypatch.setenv("COLUMNS", "220")
    outcome = cli.run("endpoints", "some/model", tty=True)
    assert outcome.code == 0
    assert "cheap" in outcome.stdout and "pricey" in outcome.stdout
