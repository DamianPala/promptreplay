"""`report`: resolving RUN (a path, a trace name, or 'latest') and the --md side effect."""

from __future__ import annotations

from pathlib import Path

from provibench.bench.replay import ReplayOptions, ReplayResult, write_run
from provibench.bench.targets import RunSpec, Target
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli


def _write_run(runs_dir: Path, trace: str, created: str) -> Path:
    """A real run directory (via `write_run`), then renamed to a chosen `created` stamp.

    `write_run` always stamps `datetime.now(UTC)`; renaming afterwards is the simplest way
    to get two runs with a controlled, comparable ordering.
    """
    target = Target(
        name="t", url="https://x.test/v1/messages", api_key_env="X_KEY", kind="anthropic"
    )
    spec = RunSpec(target=target, model="model-a")
    result = ReplayResult(
        seq=1,
        turn=1,
        status=200,
        latency_ms=1.0,
        message_id="m1",
        model="model-a",
        prompt_total=10,
        cached=0,
        cache_write=0,
        output_tokens=1,
    )
    run_dir = write_run(
        runs_dir, trace, "c1", [spec], ReplayOptions(), results={spec.label: [result]}
    )
    renamed = run_dir.parent / created
    run_dir.rename(renamed)
    return renamed


def test_report_accepts_a_run_directory_path_directly(cli: Cli, bench_paths: BenchPaths) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    outcome = cli.run("report", str(run_dir), env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    doc = outcome.document
    assert doc["run_dir"] == str(run_dir)
    assert doc["trace"] == "t"
    assert doc["conversation"] == "c1"
    assert doc["markdown_path"] is None
    assert doc["changed"] is False
    [summary] = [d for d in map(as_document, as_list(doc["summaries"]) or []) if d]
    assert summary["label"] == "t:model-a"


def test_report_trace_name_picks_the_newest_run_under_it(cli: Cli, bench_paths: BenchPaths) -> None:
    _write_run(bench_paths.runs_dir, "t", "20250101-000000")
    newest = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    outcome = cli.run("report", "t", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["run_dir"] == str(newest)


def test_report_latest_picks_the_newest_run_across_every_trace(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    _write_run(bench_paths.runs_dir, "t1", "20250101-000000")
    newest = _write_run(bench_paths.runs_dir, "t2", "20260101-000000")
    outcome = cli.run("report", "latest", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["run_dir"] == str(newest)


def test_report_unknown_run_is_not_found(cli: Cli, bench_paths: BenchPaths) -> None:
    outcome = cli.run("report", "nope", env=bench_paths.env)
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"

    empty_runs_dir = cli.run("report", "latest", env=bench_paths.env)
    assert empty_runs_dir.code == 1
    assert empty_runs_dir.error["kind"] == "not_found"


def test_report_md_writes_markdown_and_reports_changed(cli: Cli, bench_paths: BenchPaths) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    md_path = cli.root / "out" / "report.md"
    outcome = cli.run("report", str(run_dir), "--md", str(md_path), env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["markdown_path"] == str(md_path)
    assert outcome.document["changed"] is True
    assert md_path.is_file()
    assert md_path.read_text(encoding="utf-8").startswith("| label |")


def test_report_md_relative_path_resolves_against_injected_cwd(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    outcome = cli.run("report", str(run_dir), "--md", "relative-report.md", env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert (cli.root / "relative-report.md").is_file()
    assert outcome.document["markdown_path"] == str(cli.root / "relative-report.md")
