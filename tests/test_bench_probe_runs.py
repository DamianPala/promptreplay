"""Tests for promptreplay.bench.probe_runs: persisting and reading a probe run."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptreplay.bench.estimate import SpecPrices
from promptreplay.bench.probe_models import ProbeOptions, ProbeResult, ProbeRun
from promptreplay.bench.probe_runs import load_probe_run, write_probe_run
from promptreplay.bench.probe_summary import summarize_probe
from promptreplay.bench.targets import Prices, RunSpec, Target


def _spec(model: str = "model-a") -> RunSpec:
    target = Target(
        name="fake", url="https://api.test/v1/messages", api_key_env="FAKE_KEY", kind="anthropic"
    )
    return RunSpec(target=target, model=model, providers=[])


def _precheck_record(label: str, *, status: int = 200) -> ProbeResult:
    return ProbeResult(
        spec_label=label,
        rung=1,
        role="precheck",
        attempt=0,
        seq=1,
        status=status,
        latency_ms=5.0,
        prompt_total=100,
        cached=0,
    )


def _prices() -> SpecPrices:
    return SpecPrices(
        prices=Prices(input=1.0, cache_read=0.1, cache_write=1.0, output=2.0), source="table"
    )


def test_precheck_requests_are_persisted_apart_from_any_specs_own_file(tmp_path: Path) -> None:
    """Item 5: the availability check's own requests land in `precheck.jsonl`, never inside a
    spec's own file -- so a candidate the check dropped never enters a spec's own records."""
    spec = _spec()
    run = ProbeRun(
        run_hex="abc123def456",
        options=ProbeOptions(rungs=[1], repeats=[1]),
        records={spec.label: []},
    )
    precheck = [
        _precheck_record("candidate:dropped"),
        _precheck_record("candidate:dropped", status=404),
    ]
    prices = {"candidate:dropped": _prices()}

    run_dir = write_probe_run(
        tmp_path, "t", "c1", run, [spec], prices=prices, precheck=precheck, trace_prompt_tokens=500
    )

    assert (run_dir / "precheck.jsonl").is_file()
    persisted = (run_dir / "precheck.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(persisted) == 2

    meta, records = load_probe_run(run_dir)
    assert meta.precheck is not None
    assert meta.precheck.requests == 2
    # one served at the listed input price (1.0 $/M over 100 tokens), one refused and free
    assert meta.precheck.spend_usd == pytest.approx(1.0e-4)
    assert meta.trace_prompt_tokens == 500
    # never folded into any spec's own records, so it never enters that spec's own numbers
    assert records[spec.label] == []
    summary = summarize_probe(spec.label, records[spec.label])
    assert summary.errors == 0
    assert summary.notes == []


def test_a_run_with_no_availability_check_writes_no_precheck_file(tmp_path: Path) -> None:
    """The common case -- no `--top`, so nothing was checked -- leaves `precheck` `None`
    and writes no `precheck.jsonl`, matching a run written before this field existed."""
    spec = _spec()
    run = ProbeRun(
        run_hex="abc123def456",
        options=ProbeOptions(rungs=[1], repeats=[1]),
        records={spec.label: []},
    )

    run_dir = write_probe_run(tmp_path, "t", "c1", run, [spec])

    assert not (run_dir / "precheck.jsonl").exists()
    meta, _records = load_probe_run(run_dir)
    assert meta.precheck is None
    assert meta.trace_prompt_tokens is None
    assert meta.listing_prices is None


def test_precheck_worst_case_is_persisted_alongside_its_spend(tmp_path: Path) -> None:
    """O4: the pre-check's own worst case is persisted next to its spend, so `report` can
    fold it into the run's total without needing `precheck.jsonl`'s raw records again."""
    spec = _spec()
    run = ProbeRun(
        run_hex="abc123def456",
        options=ProbeOptions(rungs=[1], repeats=[1]),
        records={spec.label: []},
    )
    precheck = [_precheck_record("candidate:dropped")]
    prices = {"candidate:dropped": _prices()}

    run_dir = write_probe_run(tmp_path, "t", "c1", run, [spec], prices=prices, precheck=precheck)

    meta, _records = load_probe_run(run_dir)
    assert meta.precheck is not None
    # one served request, 100 uncached prompt tokens at the listed 1.0 $/M input rate
    assert meta.precheck.worst_case_usd == pytest.approx(1.0e-4)


def test_listing_prices_round_trips_and_reprices_the_unpinned_spec_by_it(
    tmp_path: Path,
) -> None:
    """O1: a run's own listing, once persisted, reprices an unpinned spec from it instead of
    fitting the served provider's billed records -- on a freshly written run, not only on
    the real fixture that predates the field."""
    spec = RunSpec(
        target=Target(
            name="or",
            url="https://openrouter.test/v1/messages",
            api_key_env="OR_KEY",
            kind="openrouter",
        ),
        model="model-a",
        providers=[],
    )
    served = ProbeResult(
        spec_label=spec.label,
        rung=1,
        role="warm",
        attempt=1,
        seq=1,
        status=200,
        latency_ms=10.0,
        provider="GMICloud",
        prompt_total=100,
        cached=0,
    )
    run = ProbeRun(
        run_hex="abc123def456",
        options=ProbeOptions(rungs=[1], repeats=[1]),
        records={spec.label: [served]},
    )
    listing = {"gmicloud": Prices(input=0.285, cache_read=0.0057, cache_write=0.0, output=1.14)}

    run_dir = write_probe_run(tmp_path, "t", "c1", run, [spec], listing_prices=listing)

    meta, records = load_probe_run(run_dir)
    assert meta.listing_prices == listing
    summary = summarize_probe(
        spec.label,
        records[spec.label],
        unpinned_gateway=True,
        listing=meta.listing_prices,
    )
    assert summary.priced_as == "served: GMICloud (listed)"
    assert summary.listed_input == 0.285
    assert summary.listed_source == "listing"
