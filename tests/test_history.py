"""`history` and `compare`: the time axis over a runs directory.

The run directories are built with the writers the commands themselves use
(`write_probe_run`, `write_run`) and then stamped with a fixed date, because "a day apart"
is the thing these commands exist to show and a real clock cannot produce three of them.
The stamps sit around the `FakeClock`'s start, so `--since` has a settled age to filter on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from provibench.bench.estimate import SpecPrices
from provibench.bench.probe_models import ProbeOptions, ProbeResult, ProbeRun, Role
from provibench.bench.probe_runs import EndpointSnapshot, load_probe_run, write_probe_run
from provibench.bench.replay import ReplayOptions, ReplayResult, write_run
from provibench.bench.targets import Prices, RunSpec, Target
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli

_NOW = 1_800_000_000.0
"""The `FakeClock`'s start (2027-01-15T08:00Z), so a fixture's age is what the clock says."""
_DAY = 86_400.0
_MODEL = "model"
_TARGET = Target(
    name="or", url="https://openrouter.test/v1/messages", api_key_env="OR_KEY", kind="openrouter"
)


@dataclass(frozen=True)
class Spec:
    """One spec in a synthetic run: which endpoint, what its warm reads found, and its price.

    A `None` in `cached` is a warm read that failed: it scores nothing in the hit rate and
    is one error, which is the difference between "the provider missed" and "the provider
    did not answer".
    """

    tag: str
    cached: tuple[int | None, ...] = (100, 100)
    price_input: float = 0.3
    price_cache_read: float = 0.03
    ttft_ms: float = 400.0
    gen_tok_s: float = 40.0
    quantization: str = "fp8"
    uptime_1d: float = 99.5


def spec(tag: str, **overrides: Any) -> Spec:
    return Spec(tag=tag, **overrides)


def run_spec(tag: str) -> RunSpec:
    return RunSpec(target=_TARGET, model=_MODEL, providers=[tag])


def stamp_at(days_ago: float) -> str:
    """A `created` stamp for a run that many days before the fake clock's now."""
    return datetime.fromtimestamp(_NOW - days_ago * _DAY, UTC).strftime("%Y%m%d-%H%M%S")


def date_at(days_ago: float) -> str:
    """The date `stamp_at` renders as in a table."""
    return datetime.fromtimestamp(_NOW - days_ago * _DAY, UTC).strftime("%Y-%m-%d")


def write_probe_fixture(
    runs_dir: Path, trace: str, created: str, specs: list[Spec], *, snapshot: bool = True
) -> Path:
    """One probe run directory, dated: a cold write, the warm reads, and a streamed request."""
    records: dict[str, list[ProbeResult]] = {}
    for item in specs:
        label = run_spec(item.tag).label
        records[label] = [
            _record(label, "cold", attempt=0, cached=0, cache_write=100),
            *[
                _record(label, "warm", attempt=index, cached=hit)
                for index, hit in enumerate(item.cached, start=1)
            ],
            _record(label, "stream", attempt=0, ttft_ms=item.ttft_ms, gen_tok_s=item.gen_tok_s),
        ]
    run = ProbeRun(
        run_hex="abc123def456", options=ProbeOptions(rungs=[1], repeats=[2]), records=records
    )
    run_dir = write_probe_run(
        runs_dir,
        trace,
        "c1",
        run,
        [run_spec(item.tag) for item in specs],
        endpoints=(
            {
                run_spec(item.tag).label: EndpointSnapshot(
                    tag=item.tag,
                    price_input=item.price_input,
                    price_cache_read=item.price_cache_read,
                    source="openrouter-endpoint",
                    quantization=item.quantization,
                    context_length=128000,
                    uptime_1d=item.uptime_1d,
                    status=0,
                )
                for item in specs
            }
            if snapshot
            else {}
        ),
        prices={run_spec(item.tag).label: _prices(item) for item in specs},
    )
    return _stamp(run_dir, created)


def write_full_fixture(runs_dir: Path, trace: str, created: str, tag: str) -> Path:
    """One full-replay run directory, dated: two turns, the second of them half cached."""
    run = run_spec(tag)
    results = [
        _turn(1, cached=0, provider="Novita"),
        _turn(2, cached=50, provider="Novita"),
    ]
    run_dir = write_run(runs_dir, trace, "c1", [run], ReplayOptions(), results={run.label: results})
    return _stamp(run_dir, created)


def _turn(turn: int, *, cached: int, provider: str) -> ReplayResult:
    return ReplayResult(
        seq=turn,
        turn=turn,
        status=200,
        latency_ms=1.0,
        message_id=f"m{turn}",
        model=_MODEL,
        provider=provider,
        prompt_total=100,
        cached=cached,
        cache_write=0,
        output_tokens=1,
    )


def _record(
    label: str,
    role: Role,
    *,
    attempt: int,
    cached: int | None = 0,
    cache_write: int = 0,
    ttft_ms: float | None = None,
    gen_tok_s: float | None = None,
) -> ProbeResult:
    failed = cached is None
    return ProbeResult(
        spec_label=label,
        rung=1,
        role=role,
        attempt=attempt,
        seq=attempt + 1,
        status=500 if failed else 200,
        latency_ms=40.0,
        ttft_ms=ttft_ms,
        gen_tok_s=gen_tok_s,
        prompt_total=100,
        cached=0 if failed else cached,
        cache_write=cache_write,
        error="boom" if failed else None,
    )


def _prices(item: Spec) -> SpecPrices:
    return SpecPrices(
        prices=Prices(
            input=item.price_input,
            cache_read=item.price_cache_read,
            cache_write=item.price_input,
            output=0.6,
        ),
        source="openrouter-endpoint",
        provider=item.tag.title(),
    )


def _stamp(run_dir: Path, created: str) -> Path:
    """Date a run directory: its `created` field and the directory name are the same fact."""
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created"] = created
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    stamped = run_dir.parent / created
    run_dir.rename(stamped)
    return stamped


def three_runs(bench_paths: BenchPaths) -> list[Path]:
    """Three runs a day apart for two specs: the oldest missed, the newest cached fully."""
    return [
        write_probe_fixture(
            bench_paths.runs_dir,
            "sample",
            stamp_at(3.0),
            [spec("novita", cached=(0, 0)), spec("relace")],
        ),
        write_probe_fixture(
            bench_paths.runs_dir,
            "sample",
            stamp_at(2.0),
            [spec("novita", cached=(100, 0)), spec("relace", cached=(0, 100), price_input=0.2)],
        ),
        write_probe_fixture(
            bench_paths.runs_dir,
            "sample",
            stamp_at(1.0),
            [
                spec("novita", cached=(100, 100), price_input=0.24),
                spec("relace", cached=(100, 100), price_input=0.2),
            ],
        ),
    ]


def rows_of(document: dict[str, object]) -> list[dict[str, object]]:
    return [d for d in map(as_document, as_list(document["rows"]) or []) if d]


def run_ref(document: dict[str, object], key: str) -> dict[str, object]:
    """One of `compare`'s two run refs, for a test that asserts which run it resolved."""
    return as_document(document[key]) or {}


def test_history_renders_three_runs_a_day_apart_as_one_table(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    three_runs(bench_paths)
    outcome = cli.run("history", tty_stdout=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    lines = outcome.stdout.splitlines()
    assert lines[0] == "specs: or:model@<provider>"
    assert "date" in lines[1] and lines[1].rstrip().endswith("in $/M")
    dated = [line for line in lines if line.startswith(date_at(3.0)[:4])]
    assert len(dated) == 6  # three runs of two specs
    # one provider at a time, oldest first within it
    assert [line.split(" | ")[1].strip() for line in dated] == ["@novita"] * 3 + ["@relace"] * 3
    assert [line.split(" | ")[0].strip().split()[0] for line in dated[:3]] == [
        date_at(3.0),
        date_at(2.0),
        date_at(1.0),
    ]
    series = [line for line in lines if line.endswith("3 run(s)")]
    assert len(series) == 2  # one sparkline per spec
    assert series[0].startswith("@novita") and "▁▅█" in series[0]  # oldest on the left
    assert "█" in series[1]


def test_history_json_carries_the_rows_and_the_series(cli: Cli, bench_paths: BenchPaths) -> None:
    three_runs(bench_paths)
    document = cli.run("history", "--json", env=bench_paths.env).document
    assert document["runs"] == 3
    rows = rows_of(document)
    assert len(rows) == 6
    [newest] = [
        row for row in rows if row["spec"] == "or:model@novita" and row["created"] == stamp_at(1)
    ]
    assert newest["hit_rate"] == 1.0
    assert newest["listed_input"] == 0.24
    assert newest["listed_source"] == "openrouter-endpoint"
    assert newest["quantization"] == "fp8" and newest["status"] == "0"
    series = [d for d in map(as_document, as_list(document["series"]) or []) if d]
    assert [entry["spec"] for entry in series] == ["or:model@novita", "or:model@relace"]
    assert series[0]["hit_rates"] == [0.0, 0.5, 1.0]
    assert series[1]["runs"] == 3


def test_history_since_drops_the_oldest_run(cli: Cli, bench_paths: BenchPaths) -> None:
    three_runs(bench_paths)
    document = cli.run("history", "--since", "2d", "--json", env=bench_paths.env).document
    assert document["runs"] == 2
    rows = rows_of(document)
    assert len(rows) == 4
    assert {row["created"] for row in rows} == {stamp_at(2), stamp_at(1)}


def test_history_rejects_a_duration_it_cannot_read(cli: Cli, bench_paths: BenchPaths) -> None:
    three_runs(bench_paths)
    outcome = cli.run("history", "--since", "2w", env=bench_paths.env)
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"


def test_history_filters_by_model_and_names_the_ones_it_has(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    three_runs(bench_paths)
    by_slug = cli.run("history", _MODEL, "--json", env=bench_paths.env)
    assert by_slug.code == 0, by_slug.stderr
    assert len(rows_of(by_slug.document)) == 6
    by_head = cli.run("history", f"or:{_MODEL}", "--json", env=bench_paths.env)
    assert by_head.document == by_slug.document

    missing = cli.run("history", "other-model", env=bench_paths.env)
    assert missing.code == 1
    assert missing.error["kind"] == "not_found"
    assert _MODEL in str(missing.error["hint"])


def test_history_with_no_runs_is_not_found(cli: Cli, bench_paths: BenchPaths) -> None:
    outcome = cli.run("history", env=bench_paths.env)
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"


def test_history_keeps_a_full_replay_with_dashes_for_what_it_has_not_measured(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    write_full_fixture(bench_paths.runs_dir, "sample", stamp_at(1.0), "novita")
    document = cli.run("history", "--json", env=bench_paths.env).document
    [row] = rows_of(document)
    assert row["protocol"] == "full"
    assert row["hit_rate"] == 0.5
    assert row["ttft_ms"] is None and row["gen_tok_s"] is None
    assert row["listed_input"] is None and row["listed_source"] == "n/a"


def test_compare_latest_previous_prints_the_delta_per_spec(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    three_runs(bench_paths)
    outcome = cli.run("compare", "previous", "latest", tty_stdout=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    lines = outcome.stdout.splitlines()
    assert lines[0].startswith("A: ") and lines[0].endswith("probe")
    assert date_at(2.0) in lines[0] and date_at(1.0) in lines[1]
    novita = next(line for line in lines if line.startswith("@novita"))
    assert "50.0 → 100.0 (+50.0)" in novita  # hit rate in per cent, the delta in points
    assert "0.165 → 0.030 (-0.135)" in novita  # eff $/M: half the reads hit, then all of them
    listed = next(index for index, line in enumerate(lines) if "listed $/M in" in line)
    assert "0.300 → 0.240 (-0.060)" in lines[listed + 2]
    assert "0.030 → 0.030 (+0.000)" in lines[listed + 2]


def test_compare_reads_the_arguments_as_a_and_b(cli: Cli, bench_paths: BenchPaths) -> None:
    """The delta is B minus A of the arguments given, whichever way round they are named."""
    three_runs(bench_paths)
    document = cli.run("compare", "latest", "previous", "--json", env=bench_paths.env).document
    assert run_ref(document, "run_a")["created"] == stamp_at(1)  # latest
    assert run_ref(document, "run_b")["created"] == stamp_at(2)  # previous
    [row] = [r for r in rows_of(document) if r["spec"] == "or:model@novita"]
    hit = as_document(row["hit_rate"]) or {}
    assert (hit["a"], hit["b"], hit["delta"]) == (1.0, 0.5, -0.5)  # the change is backwards


def test_compare_json_pairs_the_metrics_and_names_unpaired_specs(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    write_probe_fixture(
        bench_paths.runs_dir, "sample", stamp_at(1.0), [spec("novita"), spec("relace")]
    )
    write_probe_fixture(
        bench_paths.runs_dir,
        "sample",
        stamp_at(0.5),
        [spec("novita", cached=(None, None)), spec("gmicloud")],
    )
    document = cli.run("compare", "previous", "latest", "--json", env=bench_paths.env).document
    assert run_ref(document, "run_a")["created"] == stamp_at(1)
    assert run_ref(document, "run_b")["created"] == stamp_at(0.5)
    rows = rows_of(document)
    assert [row["spec"] for row in rows] == ["or:model@novita"]
    hit = as_document(rows[0]["hit_rate"]) or {}
    assert (hit["a"], hit["b"], hit["delta"]) == (1.0, None, None)
    assert document["only_in_a"] == ["or:model@relace"]
    assert document["only_in_b"] == ["or:model@gmicloud"]
    assert as_document(rows[0]["gen_tok_s"]) == {"a": 40.0, "b": 40.0, "delta": 0.0}


def test_compare_refuses_two_traces_until_forced(cli: Cli, bench_paths: BenchPaths) -> None:
    first = write_probe_fixture(bench_paths.runs_dir, "one", stamp_at(1.0), [spec("novita")])
    second = write_probe_fixture(bench_paths.runs_dir, "two", stamp_at(0.5), [spec("novita")])

    outcome = cli.run("compare", str(first), str(second), env=bench_paths.env)
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "different traces" in str(outcome.error["message"])
    assert "--force" in str(outcome.error["hint"])

    forced = cli.run("compare", str(first), str(second), "--force", "--json", env=bench_paths.env)
    assert forced.code == 0, forced.stderr
    assert run_ref(forced.document, "run_a")["trace"] == "one"
    assert run_ref(forced.document, "run_b")["trace"] == "two"
    assert rows_of(forced.document)


def test_compare_refuses_a_probe_against_a_full_replay(cli: Cli, bench_paths: BenchPaths) -> None:
    probe = write_probe_fixture(bench_paths.runs_dir, "sample", stamp_at(2.0), [spec("novita")])
    full = write_full_fixture(bench_paths.runs_dir, "sample", stamp_at(1.0), "novita")

    outcome = cli.run("compare", str(probe), str(full), env=bench_paths.env)
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"
    assert "different protocols" in str(outcome.error["message"])


def test_compare_without_a_run_before_the_newest_is_not_found(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    write_probe_fixture(bench_paths.runs_dir, "sample", stamp_at(1.0), [spec("novita")])
    outcome = cli.run("compare", "latest", "previous", env=bench_paths.env)
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"


def test_an_old_run_without_the_snapshot_shows_no_listed_price(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    old = write_probe_fixture(
        bench_paths.runs_dir, "sample", stamp_at(2.0), [spec("novita")], snapshot=False
    )
    newer = write_probe_fixture(
        bench_paths.runs_dir, "sample", stamp_at(1.0), [spec("novita")], snapshot=False
    )
    document = cli.run("history", "--json", env=bench_paths.env).document
    rows = rows_of(document)
    assert [row["listed_input"] for row in rows] == [None, None]
    assert [row["listed_source"] for row in rows] == ["n/a", "n/a"]

    text = cli.run("history", tty_stdout=True, env=bench_paths.env).stdout
    oldest = next(line for line in text.splitlines() if line.startswith(date_at(2.0)))
    assert oldest.rstrip().endswith("-")  # the listed price the run never recorded

    compared = cli.run("compare", str(old), str(newer), "--json", env=bench_paths.env)
    assert compared.code == 0, compared.stderr
    assert compared.document["listed"] == []
    assert rows_of(compared.document)  # the measured numbers are still compared
    text = cli.run("compare", str(old), str(newer), tty_stdout=True, env=bench_paths.env).stdout
    assert "listed prices: neither run recorded an endpoint snapshot" in text


def test_load_probe_run_ignores_the_pre_slice_endpoint_listing(bench_paths: BenchPaths) -> None:
    run_dir = write_probe_fixture(
        bench_paths.runs_dir, "sample", stamp_at(1.0), [spec("novita")], snapshot=False
    )
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # the shape a run written before this slice stored: the gateway's whole list, by model
    meta["endpoints"] = {
        "model": [{"provider_name": "Novita", "tag": "novita", "quantization": "fp8"}]
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    loaded, records = load_probe_run(run_dir)
    assert loaded.endpoints == {}  # an endpoint listing is not a per-spec snapshot
    assert loaded.specs and records  # the run still loads
