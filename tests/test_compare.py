"""`compare`: the delta between two runs of one trace.

The run directories come from the same dated writers `tests/test_history.py` builds its
listings with, so the two commands are tested on one shape of data.
"""

from __future__ import annotations

from provibench.core.documents import as_document
from tests.conftest import BenchPaths, Cli
from tests.test_history import (
    date_at,
    rows_of,
    run_ref,
    spec,
    stamp_at,
    three_runs,
    write_full_fixture,
    write_probe_fixture,
)


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
    assert "50.0 → 100.0 (+50.0)" in novita  # pooled hit rate in per cent, the delta in points
    # eff $/M prices from each run's first read, which hit both times, so it does not move
    # even though the pooled hit rate above does
    assert "0.030 → 0.030 (+0.000)" in novita
    listed = next(index for index, line in enumerate(lines) if "listed $/M in" in line)
    assert "0.300 → 0.240 (-0.060)" in lines[listed + 2]
    assert "0.030 → 0.030 (+0.000)" in lines[listed + 2]


def test_compare_reads_the_arguments_as_a_and_b(cli: Cli, bench_paths: BenchPaths) -> None:
    """The delta is B minus A of the arguments given, whichever way round they are named."""
    three_runs(bench_paths)
    document = cli.run("compare", "latest", "previous", "--json", env=bench_paths.env).document
    assert run_ref(document, "run_a")["created"] == stamp_at(1)  # latest
    assert run_ref(document, "run_b")["created"] == stamp_at(2)  # previous
    [row] = [r for r in rows_of(document) if r["endpoint"] == "or:model@novita"]
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
    assert [row["endpoint"] for row in rows] == ["or:model@novita"]
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

    forced = cli.run(
        "compare", str(probe), str(full), "--force", tty_stdout=True, env=bench_paths.env
    )
    assert forced.code == 0, forced.stderr
    assert any("or:model@novita" in line for line in forced.stdout.splitlines())


def test_compare_without_a_run_before_the_newest_is_not_found(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    write_probe_fixture(bench_paths.runs_dir, "sample", stamp_at(1.0), [spec("novita")])
    outcome = cli.run("compare", "latest", "previous", env=bench_paths.env)
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"


def test_report_hints_latest_only_and_compare_hints_previous_too(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """`previous` is `compare`'s name, so `report`'s hint must not recommend it."""
    write_probe_fixture(bench_paths.runs_dir, "sample", stamp_at(1.0), [spec("novita")])
    report = cli.run("report", "previous", env=bench_paths.env)
    assert report.code == 1 and report.error["kind"] == "not_found"
    assert "'latest'" in str(report.error["hint"])
    assert "previous" not in str(report.error["hint"])
    compare = cli.run("compare", "nope", "latest", env=bench_paths.env)
    assert compare.code == 1 and compare.error["kind"] == "not_found"
    assert "'previous'" in str(compare.error["hint"])
