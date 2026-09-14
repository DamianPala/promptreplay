"""`bench.selection` and the availability reasons: the pure part of a sweep's choice.

These are the criteria read on their own — the floor, the ranking keys, the ZDR
intersection, the cost of the pre-check and the words a failed candidate is reported in —
so a change to what "best" means shows up here rather than in a table.
"""

from __future__ import annotations

import json

import pytest

from provibench.bench.estimate import SpecPrices, precheck_cost
from provibench.bench.openrouter import Endpoint
from provibench.bench.precheck import unavailable_reason
from provibench.bench.probe_models import ProbeOptions
from provibench.bench.selection import (
    UPTIME_FLOOR,
    SelectionDrop,
    SweepInfo,
    missing_percentiles,
    pinned_spec,
    rank_candidates,
    ranked,
    render_candidates,
    select_candidates,
    sort_key,
    stability_reason,
)
from provibench.bench.targets import Prices, RunSpec, Target
from provibench.bench.trace import RecordedResponse, TraceEntry, Usage

_GATEWAY = Target(
    name="or", url="https://openrouter.test/v1/messages", api_key_env="OR_KEY", kind="openrouter"
)
_MODEL = "deepseek/deepseek-v4.1-flash"

_GUARDRAIL_MESSAGE = (
    "0 endpoints out of 1 requested are available matching your guardrail restrictions and "
    "data policy. We removed them for the following reasons (an endpoint may have matched "
    "multiple reasons):\nPaid model training violation (account settings): 1 endpoint "
    "excluded; configurable at https://openrouter.ai/settings/privacy"
)


def _endpoint(
    tag: str,
    *,
    price: float = 1.0,
    uptime: float | None = 99.9,
    status: int | str | None = 0,
    latency: float | None = None,
    throughput: float | None = None,
) -> Endpoint:
    return Endpoint(
        provider_name=tag.split("/")[0].title(),
        tag=tag,
        prices=Prices(input=price, cache_read=price, cache_write=price, output=price),
        uptime_1d=uptime,
        status=status,
        latency_ms_30m=latency,
        throughput_30m=throughput,
    )


def _tags(endpoints: list[Endpoint], sort: str) -> list[str]:
    return [endpoint.tag for endpoint in rank_candidates(endpoints, sort)]


def _entry(prompt_tokens: int) -> TraceEntry:
    return TraceEntry(
        seq=1,
        ts="2026-01-01T00:00:00Z",
        path="/v1/messages",
        body={"messages": [{"role": "user", "content": "hi"}]},
        conversation="c1",
        response=RecordedResponse(
            status=200, latency_ms=1.0, usage=Usage(input_tokens=prompt_tokens)
        ),
    )


def test_stability_reason_names_the_status_and_the_uptime() -> None:
    assert stability_reason(_endpoint("a"), included=False) is None
    assert stability_reason(_endpoint("a", status=-2), included=False) == "status -2"
    assert stability_reason(_endpoint("a"), included=True) is None
    # the API sends the status as a number or as text; both read as the same floor
    assert stability_reason(_endpoint("a", status="-2"), included=False) == "status -2"
    assert stability_reason(_endpoint("a", status="-2"), included=True) is None
    assert (
        stability_reason(_endpoint("a", uptime=UPTIME_FLOOR - 0.1), included=False)
        == "uptime 1d 96.90 %"
    )
    assert stability_reason(_endpoint("a", uptime=UPTIME_FLOOR), included=False) is None


def test_a_drop_reason_next_to_the_floor_does_not_round_up_to_it() -> None:
    """96.96 at one decimal reads `97.0 %`: the reason would contradict the criterion."""
    assert stability_reason(_endpoint("a", uptime=96.96), included=False) == "uptime 1d 96.96 %"


def test_stability_reason_keeps_what_the_api_did_not_say() -> None:
    """An endpoint with no status and no uptime is not degraded, it is unmeasured."""
    assert stability_reason(_endpoint("a", status=None, uptime=None), included=False) is None
    assert stability_reason(_endpoint("a", status="unknown"), included=False) is None


def test_price_ties_break_by_throughput_then_uptime() -> None:
    endpoints = [
        _endpoint("unknown", price=1.0, throughput=None),
        _endpoint("slow", price=1.0, throughput=10.0),
        _endpoint("fast", price=1.0, throughput=90.0),
        _endpoint("fast-shaky", price=1.0, throughput=90.0, uptime=98.0),
    ]
    assert _tags(endpoints, "price") == ["fast", "fast-shaky", "slow", "unknown"]


def test_uptime_sorts_descending_and_ties_on_price() -> None:
    endpoints = [
        _endpoint("expensive", price=2.0, uptime=99.0),
        _endpoint("cheap", price=1.0, uptime=99.0),
        _endpoint("best", price=9.0, uptime=100.0),
    ]
    assert _tags(endpoints, "uptime") == ["best", "cheap", "expensive"]


def test_throughput_and_latency_sort_by_their_percentile() -> None:
    endpoints = [
        _endpoint("a", throughput=10.0, latency=300.0),
        _endpoint("b", throughput=50.0, latency=100.0),
        _endpoint("c", throughput=None, latency=None),
    ]
    assert _tags(endpoints, "throughput") == ["b", "a", "c"]
    assert _tags(endpoints, "latency") == ["b", "a", "c"]


def test_an_unknown_sort_key_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="unknown sort key"):
        sort_key(_endpoint("a"), "cheapest")
    assert sort_key(_endpoint("a"), "price")


def test_missing_percentiles_lists_the_tags_without_one() -> None:
    endpoints = [_endpoint("a", latency=1.0), _endpoint("b")]
    assert missing_percentiles(endpoints, "latency") == ["b"]
    assert missing_percentiles(endpoints, "throughput") == ["a", "b"]


def test_select_candidates_drops_non_zdr_first_then_the_floor() -> None:
    endpoints = [
        _endpoint("novita", price=0.3),
        _endpoint("relace/fp4", price=0.1, status=-2),
        _endpoint("siliconflow", price=0.2, uptime=80.0),
        _endpoint("gmicloud", price=0.4),
    ]
    selection = select_candidates(
        endpoints,
        gateway=_GATEWAY,
        model=_MODEL,
        sort="price",
        zdr=True,
        zdr_tags={"novita", "relace/fp4", "gmicloud"},
    )
    assert [endpoint.tag for endpoint in selection.kept] == ["novita", "gmicloud"]
    assert [(drop.tag, drop.reason) for drop in selection.dropped] == [
        ("relace/fp4", "status -2"),
        ("siliconflow", "not ZDR"),
    ]
    # a drop carries the run-spec label of the endpoint it removes, not just its tag
    assert selection.dropped[1].spec == f"or:{_MODEL}@siliconflow"
    assert selection.dropped[1].checked is False


def test_candidate_table_prints_two_decimals_only_next_to_the_floor() -> None:
    """A kept uptime that rounds to the floor gets its second digit; a healthy one does not."""
    selection = select_candidates(
        [_endpoint("parasail/fp8", uptime=96.96), _endpoint("novita", uptime=99.9)],
        gateway=_GATEWAY,
        model=_MODEL,
        include=["parasail"],
    )
    lines = render_candidates(selection, model=_MODEL, sort="price", zdr=False)
    row = next(line for line in lines if line.startswith("parasail/fp8"))
    assert "96.96" in row
    healthy = next(line for line in lines if line.startswith("novita"))
    assert "99.9" in healthy and "99.90" not in healthy


def test_select_candidates_include_overrides_the_floor_for_that_tag() -> None:
    endpoints = [_endpoint("relace/fp4", status=-2), _endpoint("novita")]
    selection = select_candidates(endpoints, gateway=_GATEWAY, model=_MODEL, include=["relace"])
    assert [endpoint.tag for endpoint in selection.kept] == ["relace/fp4", "novita"]
    assert selection.dropped == []


def test_unavailable_reason_quotes_the_first_guardrail_reason() -> None:
    """The 404 an excluded account gets: the reason the settings page would show."""
    assert (
        unavailable_reason(
            status=404, payload={"error_type": "not_found", "message": _GUARDRAIL_MESSAGE}
        )
        == "unavailable for this key: Paid model training violation (account settings)"
    )
    # the same message at the top level, and without a newline to move the reason to
    inline = {"message": _GUARDRAIL_MESSAGE.replace("\n", " ")}
    assert (
        unavailable_reason(status=404, payload=inline)
        == "unavailable for this key: Paid model training violation (account settings)"
    )
    # a 404 with no message of its own still says what the API called it
    assert unavailable_reason(status=404, payload={"error_type": "not_found"}) == (
        "unavailable for this key: not_found"
    )


def test_unavailable_reason_names_overloads_and_bare_failures() -> None:
    overloaded = {"error": {"type": "rate_limit_exceeded", "message": "slow down"}}
    assert unavailable_reason(status=429, payload=overloaded) == "unavailable: rate_limit_exceeded"
    assert unavailable_reason(status=503, payload=None) == "unavailable: HTTP 503"
    assert unavailable_reason(status=0, payload=None, error="connection reset") == (
        "unavailable: connection reset"
    )


def test_unavailable_reason_reads_the_wrapped_error_the_record_kept() -> None:
    """A 429 the gateway wraps: the failure is named inside the envelope, not by its `type`.

    The record's error is the wrapped object — the same string the probe's own
    `skipped, <error_type>` note is read from — so both lines name the refusal alike.
    """
    body: dict[str, object] = {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Provider returned error",
            "error_type": "rate_limit_exceeded",
        },
    }
    assert (
        unavailable_reason(status=429, payload=body, error=json.dumps(body["error"]))
        == "unavailable: rate_limit_exceeded"
    )
    # an envelope that only spells the wrapper: the status is the one thing that is true
    assert (
        unavailable_reason(status=429, payload={"type": "error", "message": "nope"})
        == "unavailable: HTTP 429"
    )


def test_the_recorded_block_keeps_the_status_as_text() -> None:
    """The document's `status` is one type; the API's number is the text `endpoints` prints."""
    info = SweepInfo(
        model=_MODEL,
        target="or",
        dropped=[SelectionDrop.for_spec(pinned_spec(_GATEWAY, _MODEL, "a"), "not ZDR")],
        ranking=[ranked(_endpoint("b", status=0)), ranked(_endpoint("c", status=None))],
    )
    assert info.to_document()["dropped"] == [
        {"tag": "a", "spec": f"or:{_MODEL}@a", "reason": "not ZDR", "checked": False}
    ]
    assert info.to_document()["sort"] == "price"
    # the number the API sent becomes its own text, and a status it never sent stays null
    assert info.to_document()["ranking"] == [
        {
            "tag": "b",
            "price_input": 1.0,
            "uptime_1d": 99.9,
            "status": "0",
            "latency_p50_ms": None,
            "throughput_p50_tok_s": None,
        },
        {
            "tag": "c",
            "price_input": 1.0,
            "uptime_1d": 99.9,
            "status": None,
            "latency_p50_ms": None,
            "throughput_p50_tok_s": None,
        },
    ]
    assert [row.status for row in info.ranking] == [0, None]


def test_precheck_cost_is_one_smallest_rung_per_candidate() -> None:
    """The phase's price is the smallest rung's own tokens, at each candidate's listed price."""
    entries = [_entry(100), _entry(200), _entry(300)]
    options = ProbeOptions(rungs=[2], repeats=[1]).resolved(entries)
    specs = [
        RunSpec(target=_GATEWAY, model=_MODEL, providers=["novita"]),
        RunSpec(target=_GATEWAY, model=_MODEL, providers=["siliconflow"]),
    ]
    prices = {
        f"or:{_MODEL}@novita": _prices(0.3),
        f"or:{_MODEL}@siliconflow": _prices(0.2),
    }
    cost = precheck_cost(specs, entries, options, prices)
    assert cost.requests == 2
    assert cost.tokens == 400  # turn 2 records 200 prompt tokens, sent once per candidate
    assert cost.usd == pytest.approx(200 * (0.3 + 0.2) * 1e-6)


def _prices(input_usd: float) -> SpecPrices:
    return SpecPrices(
        prices=Prices(input=input_usd, cache_read=0.0, cache_write=0.0, output=0.0),
        source="openrouter-endpoint",
        provider="p",
    )
