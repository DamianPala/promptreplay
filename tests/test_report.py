"""`report`: resolving RUN and writing one selected rendering to one result destination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from provibench.bench.replay import ReplayOptions, ReplayResult, write_run
from provibench.bench.targets import RunSpec, Target
from provibench.core.documents import as_document, as_list
from tests.conftest import BenchPaths, Cli


def _write_run(
    runs_dir: Path, trace: str, created: str, *, prompt_total: int = 10, cached: int = 0
) -> Path:
    """A real run directory (via `write_run`), then renamed to a chosen `created` stamp.

    `write_run` always stamps `datetime.now(UTC)`; renaming afterwards is the simplest way
    to get two runs with a controlled, comparable ordering. The token counts are settable
    because a cell's formatting — a thousands separator, a rounding — only differs from
    another renderer's at sizes the default fixture is too small to reach.
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
        prompt_total=prompt_total,
        cached=cached,
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
    assert doc["output_file"] is None
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


def test_report_output_file_writes_markdown_and_keeps_stdout_empty(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    md_path = cli.root / "out" / "report.md"
    outcome = cli.run(
        "report",
        str(run_dir),
        "--format",
        "md",
        "--output-file",
        str(md_path),
        env=bench_paths.env,
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout == ""
    assert md_path.is_file()
    assert md_path.read_text(encoding="utf-8").startswith("| label |")


def test_report_md_relative_path_resolves_against_injected_cwd(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    outcome = cli.run(
        "report",
        str(run_dir),
        "--format",
        "md",
        "--output-file",
        "relative-report.md",
        env=bench_paths.env,
    )
    assert outcome.code == 0, outcome.stderr
    assert (cli.root / "relative-report.md").is_file()
    assert outcome.stdout == ""


# --- the HTML report ----------------------------------------------------------


def test_report_output_file_writes_one_self_contained_html_file(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    html_path = cli.root / "out" / "report.html"
    outcome = cli.run(
        "report",
        str(run_dir),
        "--format",
        "html",
        "--output-file",
        str(html_path),
        env=bench_paths.env,
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout == ""
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "<script" not in text and "http://" not in text and "https://" not in text
    assert "<td>t:model-a</td>" in text


def test_report_without_output_file_reports_no_path(cli: Cli, bench_paths: BenchPaths) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    outcome = cli.run("report", str(run_dir), env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert outcome.document["output_file"] is None
    assert outcome.document["changed"] is False


def test_report_output_file_replaces_an_existing_file(cli: Cli, bench_paths: BenchPaths) -> None:
    """A report is derived from the run alone, so a repeat rewrites the same file (R1)."""
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    html_path = cli.root / "report.html"
    html_path.write_text("mine", encoding="utf-8")
    args = ["report", str(run_dir), "--format", "html", "--output-file", str(html_path)]

    outcome = cli.run(*args, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    first = html_path.read_text(encoding="utf-8")
    assert first.startswith("<!DOCTYPE html>")

    again = cli.run(*args, env=bench_paths.env)
    assert again.code == 0, again.stderr
    assert html_path.read_text(encoding="utf-8") == first


def test_report_format_json_is_the_json_document(cli: Cli, bench_paths: BenchPaths) -> None:
    """`--format json` names what `--json` selects, as the format_defaults index declares."""
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    named = cli.run("report", str(run_dir), "--format", "json", env=bench_paths.env)
    flagged = cli.run("report", str(run_dir), "--json", env=bench_paths.env)
    assert named.code == 0, named.stderr
    assert named.document == flagged.document

    both = cli.run("report", str(run_dir), "--json", "--format", "md", env=bench_paths.env)
    assert both.code == 2
    assert both.error["kind"] == "invalid_input"
    assert "--json" in str(both.error["message"]) and "--format md" in str(both.error["message"])


def test_report_formats_print_the_same_totals_for_one_run(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """Both written forms go through `summary_row`, so a cell cannot differ between them."""
    run_dir = _write_run(
        bench_paths.runs_dir, "t", "20260101-000000", prompt_total=20_000, cached=12_000
    )
    md_path, html_path = cli.root / "r.md", cli.root / "r.html"
    markdown = cli.run(
        "report",
        str(run_dir),
        "--format",
        "md",
        "--output-file",
        str(md_path),
        env=bench_paths.env,
    )
    html = cli.run(
        "report",
        str(run_dir),
        "--format",
        "html",
        "--output-file",
        str(html_path),
        env=bench_paths.env,
    )
    assert markdown.code == 0, markdown.stderr
    assert html.code == 0, html.stderr
    markdown = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "| 20,000 |" in markdown and "<td>20,000</td>" in html
    assert "| 12,000 |" in markdown and "<td>12,000</td>" in html
    assert "80000" not in markdown


def test_report_json_output_file_contains_the_success_document(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    output_path = cli.root / "report.json"
    outcome = cli.run(
        "report", str(run_dir), "--json", "--output-file", str(output_path), env=bench_paths.env
    )
    assert outcome.code == 0, outcome.stderr
    assert outcome.stdout == ""
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["output_file"] == str(output_path)


def test_report_rejects_json_and_named_format_together(cli: Cli, bench_paths: BenchPaths) -> None:
    run_dir = _write_run(bench_paths.runs_dir, "t", "20260101-000000")
    outcome = cli.run("report", str(run_dir), "--json", "--format", "html", env=bench_paths.env)
    assert outcome.code == 2
    assert outcome.error["kind"] == "invalid_input"


def test_probe_report_html_carries_the_tables_and_the_charts(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_probe_run(
        bench_paths.runs_dir,
        {
            "or:model@novita": [
                probe_record(provider="Novita", model="model"),
                probe_record(role="warm", attempt=1, cached=90, provider="Novita", model="model"),
            ],
            "or:model@gmicloud": [
                probe_record(spec_label="or:model@gmicloud", provider="GMICloud", model="model"),
                probe_record(
                    spec_label="or:model@gmicloud",
                    role="warm",
                    attempt=1,
                    cached=0,
                    provider="GMICloud",
                    model="model",
                ),
            ],
        },
    )
    html_path = cli.root / "probe.html"
    outcome = cli.run(
        "report",
        str(run_dir),
        "--format",
        "html",
        "--output-file",
        str(html_path),
        env=bench_paths.env,
    )
    assert outcome.code == 0, outcome.stderr
    text = html_path.read_text(encoding="utf-8")
    assert "specs: or:model@&lt;provider&gt;" in text
    assert "<td>@novita</td>" in text and "<td>@gmicloud</td>" in text
    assert text.count("<svg") == 2
    assert "@media (prefers-color-scheme: dark)" in text


# --- probe runs ---------------------------------------------------------------


def probe_record(**overrides: object) -> dict[str, object]:
    """One record as a probe run's jsonl carries it, with the fields the tests care about."""
    record: dict[str, object] = {
        "spec_label": "or:model@novita",
        "rung": 1,
        "role": "cold",
        "attempt": 0,
        "seq": 1,
        "status": 200,
        "latency_ms": 40.0,
        "prompt_total": 100,
        "cached": 0,
        "cache_write": 100,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    record.update(overrides)
    return record


def _write_probe_run(
    runs_dir: Path,
    records: dict[str, list[dict[str, object]]],
    *,
    created: str = "20260101-000000",
    legacy: bool = False,
) -> Path:
    """A persisted probe run: `run.json` plus one jsonl per spec.

    `legacy` writes the shape a 0.2.0-before-this-slice run has: no TTL, no throughput
    fields, no `kind` on the spec refs, no `ttl_s` in the options.
    """
    run_dir = runs_dir / "t" / created
    run_dir.mkdir(parents=True)
    specs: list[dict[str, object]] = []
    for label in records:
        slug = label.replace(":", "-").replace("/", "-").replace("@", "-")
        file_name = f"{slug}.jsonl"
        lines: list[str] = []
        for record in records[label]:
            if legacy:
                record = {
                    key: value
                    for key, value in record.items()
                    if key not in {"ttft_ms", "gen_tok_s", "fingerprint"}
                }
            lines.append(json.dumps(record))
        (run_dir / file_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        spec: dict[str, object] = {
            "label": label,
            "slug": slug,
            "target": "or",
            "model": "deepseek/deepseek-v4.1-flash",
            "providers": [label.partition("@")[2]],
            "file": file_name,
        }
        if not legacy:
            spec["kind"] = "openrouter"
        specs.append(spec)
    options: dict[str, object] = {"rungs": [1], "repeats": [1], "gap_s": 1.0, "warm": False}
    if not legacy:
        options |= {"throughput": True, "ttl_s": None}
    meta: dict[str, Any] = {
        "protocol": "probe",
        "trace": "t",
        "conversation": "c1",
        "created": created,
        "run_hex": "abc123def456",
        "options": options,
        "specs": specs,
        "prices": {
            label: {
                "prices": {"input": 0.3, "cache_read": 0.03, "cache_write": 0.3, "output": 0.6},
                "source": "openrouter-endpoint",
                "provider": label.partition("@")[2],
            }
            for label in records
        },
    }
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def test_probe_report_shortens_the_shared_prefix_into_a_caption(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_probe_run(
        bench_paths.runs_dir,
        {
            "or:model@novita": [
                probe_record(provider="Novita", model="model"),
                probe_record(role="warm", attempt=1, cached=90, prompt_total=100),
                probe_record(role="stream", ttft_ms=400.0, gen_tok_s=40.0, fingerprint="tok1 tok2"),
            ],
            "or:model@gmicloud": [
                probe_record(spec_label="or:model@gmicloud", provider="GMICloud", model="model"),
                probe_record(
                    spec_label="or:model@gmicloud",
                    role="warm",
                    attempt=1,
                    cached=90,
                    prompt_total=100,
                ),
                probe_record(
                    spec_label="or:model@gmicloud",
                    role="stream",
                    ttft_ms=600.0,
                    gen_tok_s=20.0,
                    fingerprint="tok1 tok2",
                ),
            ],
        },
    )
    plain = cli.run("report", str(run_dir), env=bench_paths.env)
    assert plain.code == 0, plain.stderr
    rendered = cli.run("report", str(run_dir), tty_stdout=True, env=bench_paths.env)
    assert rendered.code == 0, rendered.stderr
    out = rendered.stdout
    assert "specs: or:model@<provider>" in out
    # every spec keeps a distinct label instead of two 40-character clips of one string
    assert "@novita" in out and "@gmicloud" in out
    assert "TTFT ms" in out and "tok/s" in out
    assert "400" in out and "40.0" in out
    assert all(len(line) <= 120 for line in out.splitlines())


def test_probe_report_marks_provider_drift_against_the_reference(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_probe_run(
        bench_paths.runs_dir,
        {
            "or:model@novita": [
                probe_record(provider="Novita", model="model"),
                probe_record(role="warm", attempt=1, cached=90, provider="Novita", model="model"),
                probe_record(
                    role="stream", provider="Novita", model="model", fingerprint="tok1 tok2"
                ),
            ],
            # pinned to novita but served by another endpoint, with the same size and prefix
            "or:model@gmicloud": [
                probe_record(spec_label="or:model@gmicloud", provider="Somebody", model="model"),
                probe_record(
                    spec_label="or:model@gmicloud",
                    role="warm",
                    attempt=1,
                    cached=90,
                    provider="Somebody",
                    model="model",
                ),
                probe_record(
                    spec_label="or:model@gmicloud",
                    role="stream",
                    provider="Somebody",
                    model="model",
                    fingerprint="tok1 tok2",
                ),
            ],
        },
    )
    outcome = cli.run("report", str(run_dir), env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    summaries = [d for d in map(as_document, as_list(outcome.document["summaries"]) or []) if d]
    first, second = summaries
    assert first["drift"] is None and first["reference"] is True
    assert second["drift"] == "provider"
    assert second["served"] == "Somebody"
    assert second["tokens_delta_pct"] == pytest.approx(0.0)
    assert second["fingerprint_match"] is True

    rendered = cli.run("report", str(run_dir), tty_stdout=True, env=bench_paths.env)
    assert "provider" in rendered.stdout
    # the served names stay in the JSON; the table spends its width on drift instead
    run_header = rendered.stdout.split("\n\n")[0].splitlines()[0]
    assert "served" not in run_header


def test_probe_report_renders_a_run_without_stream_records(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    """A run from before this slice has no TTFT, no tok/s and no TTL: the columns show `-`."""
    run_dir = _write_probe_run(
        bench_paths.runs_dir,
        {"or:model@novita": [probe_record(), probe_record(role="warm", attempt=1, cached=90)]},
        legacy=True,
    )
    plain = cli.run("report", str(run_dir), env=bench_paths.env)
    assert plain.code == 0, plain.stderr
    [summary] = [d for d in map(as_document, as_list(plain.document["summaries"]) or []) if d]
    outcome = cli.run("report", str(run_dir), tty_stdout=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    assert as_list(summary["rungs"]) != []
    run_table = outcome.stdout.split("\n\n")[0].splitlines()
    header = [cell.strip() for cell in run_table[0].split("|")]
    assert header[:3] == ["spec", "hit %", "prefix %"]
    assert "TTFT ms" in header and "tok/s" in header and "drift" in header
    row = [cell.strip() for cell in run_table[2].split("|")]
    assert row[header.index("TTFT ms")] == "-"
    assert row[header.index("tok/s")] == "-"
    assert row[header.index("drift")] == "-"
    rung_table = outcome.stdout.split("\n\n")[1].splitlines()
    assert "ttl" not in [cell.strip() for cell in rung_table[0].split("|")]


def test_probe_report_shows_the_ttl_column_only_when_it_ran(
    cli: Cli, bench_paths: BenchPaths
) -> None:
    run_dir = _write_probe_run(
        bench_paths.runs_dir,
        {
            "or:model@novita": [
                probe_record(),
                probe_record(role="warm", attempt=1, cached=90),
                probe_record(role="ttl", attempt=60, cached=90),
                probe_record(role="ttl", attempt=300, cached=0),
            ]
        },
    )
    outcome = cli.run("report", str(run_dir), tty_stdout=True, env=bench_paths.env)
    assert outcome.code == 0, outcome.stderr
    rung_table = outcome.stdout.split("\n\n")[1].splitlines()
    header = [cell.strip() for cell in rung_table[0].split("|")]
    assert "ttl" in header
    row = [cell.strip() for cell in rung_table[2].split("|")]
    assert row[header.index("ttl")] == "60s:1 300s:0"

    marked = cli.run(
        "report",
        str(run_dir),
        "--format",
        "md",
        "--output-file",
        str(cli.root / "r.md"),
        env=bench_paths.env,
    )
    assert marked.code == 0, marked.stderr
    text = (cli.root / "r.md").read_text(encoding="utf-8")
    assert text.startswith("| spec |")
    assert "| 60s:1 300s:0 |" in text
