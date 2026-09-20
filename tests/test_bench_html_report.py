"""`render_html`: the probe and full-replay documents as one self-contained HTML file.

The documents are built the way `report` builds them — the same summary models, the same
`*_to_document` conversion — so the file under test is the file a run produces, with no run
directory and no network anywhere in the test.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from provibench.bench.estimate import SpecPrices
from provibench.bench.html_report import render_html
from provibench.bench.html_svg import MAX_SERIES, PairedBar, PairedBarRow, paired_horizontal_bars
from provibench.bench.probe_html_tables import ENDPOINT_CAPTION_HTML
from provibench.bench.probe_models import ProbeResult
from provibench.bench.probe_summary import summarize_probe
from provibench.bench.replay import ReplayResult
from provibench.bench.summary import summarize
from provibench.bench.trace import Usage
from provibench.commands.summary_view import probe_summary_to_document, summary_to_document
from provibench.core.documents import Document, as_document, as_list
from tests.test_report import probe_record

_SVG = re.compile(r"<svg\b.*?</svg>", re.S)


def _result(
    turn: int,
    *,
    cached: int = 0,
    prompt_total: int = 20_000,
    provider: str = "Novita",
    latency_ms: float = 100.0,
) -> ReplayResult:
    return ReplayResult(
        seq=turn,
        turn=turn,
        status=200,
        latency_ms=latency_ms,
        provider=provider,
        usage=Usage(),
        prompt_total=prompt_total,
        cached=cached,
        cache_write=prompt_total - cached,
        output_tokens=1,
    )


def probe_document() -> Document:
    """A probe run's document: two specs sharing a `target:model` head, and one that does not."""
    prices = {
        "or:model@novita": SpecPrices.model_validate(
            {
                "prices": {"input": 0.3, "cache_read": 0.03, "cache_write": 0.3, "output": 0.6},
                "source": "openrouter-endpoint",
                "provider": "novita",
            }
        ),
        "or:model@gmicloud": SpecPrices.model_validate(
            {
                "prices": {"input": 0.3, "cache_read": 0.03, "cache_write": 0.3, "output": 0.6},
                "source": "openrouter-endpoint",
                "provider": "gmicloud",
            }
        ),
        "evil<spec>:model-z": SpecPrices.model_validate(
            {
                "prices": {"input": 0.3, "cache_read": 0.03, "cache_write": 0.3, "output": 0.6},
                "source": "openrouter-endpoint",
                "provider": "other",
            }
        ),
    }
    records: dict[str, list[dict[str, object]]] = {
        "or:model@novita": [
            probe_record(provider="Novita", model="model", prompt_total=20_410),
            probe_record(role="warm", attempt=1, cached=19_100, prompt_total=20_410),
            probe_record(role="ttl", attempt=60, cached=19_100, prompt_total=20_410),
            probe_record(rung=2, provider="Novita", model="model", prompt_total=40_820, seq=2),
            probe_record(rung=2, role="warm", attempt=1, cached=40_000, prompt_total=40_820, seq=3),
            probe_record(role="stream", ttft_ms=420.0, gen_tok_s=41.0, fingerprint="tok"),
        ],
        "or:model@gmicloud": [
            probe_record(
                spec_label="or:model@gmicloud",
                provider="GMICloud",
                model="model",
                prompt_total=20_410,
            ),
            probe_record(
                spec_label="or:model@gmicloud",
                role="warm",
                attempt=1,
                cached=0,
                prompt_total=20_410,
            ),
            probe_record(
                spec_label="or:model@gmicloud",
                role="stream",
                ttft_ms=600.0,
                gen_tok_s=20.0,
                fingerprint="tok",
            ),
        ],
        "evil<spec>:model-z": [
            probe_record(spec_label="evil<spec>:model-z", prompt_total=9_000),
            probe_record(
                spec_label="evil<spec>:model-z",
                role="warm",
                attempt=1,
                cached=9_000,
                prompt_total=9_000,
            ),
        ],
    }
    summaries = [
        summarize_probe(
            label,
            [ProbeResult.model_validate(record) for record in entries],
            prices=prices[label],
            trace_prompt_tokens=50_000,
        )
        for label, entries in records.items()
    ]
    return {
        "run_dir": "/runs/t/20260101-000000",
        "trace": "t",
        "conversation": "c1",
        "created": "20260101-000000",
        "run_hex": "abc123def456",
        "protocol": "probe",
        "options": {
            "rungs": [1, 2],
            "repeats": [1, 1],
            "gap_s": 1.0,
            "warm": False,
            "throughput": True,
            "ttl_s": None,
        },
        "summaries": [probe_summary_to_document(summary) for summary in summaries],
        # What the run itself billed: the closing line `commands.probe_spend` builds for the
        # terminal and the page alike, so the page is tested with it present.
        "spend_usd": 0.1776,
        "worst_case_usd": 0.4851,
        # Matches the `trace_prompt_tokens=50_000` every summary above was built with, so the
        # session-cost chart's listed-price bar has the same session size the measured bar does.
        "trace_prompt_tokens": 50_000,
        "output_file": None,
        "changed": False,
    }


def replay_document() -> Document:
    """A full replay's document: two specs over four turns, one of them warm throughout."""
    results = [_result(turn) for turn in range(1, 5)]
    warm = [_result(turn, cached=5_000 * turn) for turn in range(1, 5)]
    summaries = [
        summarize("deepseek:model-a", results, [], []),
        summarize("deepseek:model-b", warm, [], []),
    ]
    return {
        "run_dir": "/runs/t/20260101-000000",
        "trace": "t",
        "conversation": "c1",
        "created": "20260101-000000",
        "run_hex": None,
        "protocol": "full",
        "options": {"max_tokens": 1, "delay_s": 0.0, "strip_thinking": False, "timeout_s": 60.0},
        "summaries": [summary_to_document(summary) for summary in summaries],
        "output_file": None,
        "changed": False,
    }


def failing_probe_document() -> Document:
    """Two endpoints sharing a `target:model` head, one with a failed second warm read:
    exercises the error-reason note (item 8) and the short `@novita`-style caveat label.
    `probe_document` has no failed reads, so this is the fixture the brief asks for."""
    failing = [
        probe_record(prompt_total=1_000),
        probe_record(role="warm", attempt=1, cached=1_000, prompt_total=1_000, seq=2),
        probe_record(role="warm", attempt=2, status=502, error="bad gateway", seq=3),
    ]
    clean = [
        probe_record(spec_label="or:model@gmicloud", provider="GMICloud", prompt_total=1_000),
        probe_record(
            spec_label="or:model@gmicloud",
            role="warm",
            attempt=1,
            cached=1_000,
            prompt_total=1_000,
            seq=2,
        ),
    ]
    summaries = [
        summarize_probe("or:model@novita", [ProbeResult.model_validate(r) for r in failing]),
        summarize_probe("or:model@gmicloud", [ProbeResult.model_validate(r) for r in clean]),
    ]
    return {
        "run_dir": "/runs/t/20260101-000000",
        "trace": "t",
        "conversation": "c1",
        "created": "20260101-000000",
        "run_hex": None,
        "protocol": "probe",
        "options": {
            "rungs": [1],
            "repeats": [2],
            "gap_s": 1.0,
            "warm": False,
            "throughput": False,
            "ttl_s": None,
        },
        "summaries": [probe_summary_to_document(summary) for summary in summaries],
        "output_file": None,
        "changed": False,
    }


def _bars(svg: str) -> int:
    return len(re.findall(r'<path class="bar ', svg))


# --- the file as a whole ------------------------------------------------------


def test_probe_document_renders_one_self_contained_page() -> None:
    html = render_html(probe_document())
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<html") == 1 and html.count("</body>") == 1
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html
    assert "xmlns" not in html
    assert "<link" not in html and "<img" not in html


def test_page_carries_the_run_and_its_scale() -> None:
    html = render_html(probe_document())
    header = html[html.index("<header") : html.index("</header>")]
    # the model comes from the reference endpoint (the first non-OpenRouter spec); the mode
    # and the run path both moved out of the header (item 13)
    assert "<h1>Prompt cache check: model, 3 endpoints</h1>" in header
    assert '<p class="sub">trace t · 2026-01-01 00:00 · run abc123def456</p>' in header
    assert "cold (nonce)" not in header
    # the fixture's run_dir is absolute; the page shows the run, never the operator's paths,
    # now in the footer rather than the header
    assert '<footer class="foot"><p class="where">runs/t/20260101-000000</p></footer>' in html
    assert "/runs/" not in html


def test_both_tables_keep_the_terminal_columns_and_short_labels() -> None:
    html = render_html(probe_document())
    # the shared prefix is not a stray spec string under the title any more: the table's own
    # caption says what the folded `@tag` labels stand for (F10).
    assert '<p class="caption">endpoints: or:model@&lt;provider&gt;</p>' not in html
    assert (
        "The two @ rows go through OpenRouter, pinned to the named provider; all serve model."
    ) in html
    endpoint_table, rung_table = html.split('<div class="scroll">')[1:3]
    assert '<th scope="col"' in endpoint_table and ">endpoint</th>" in endpoint_table
    for column in (
        "hit %",
        "1st hit %",
        "cached %",
        "in $/M",
        "cache $/M",
        "eff $/M",
        "this trace $",
        "TTFT ms",
        "tok/s",
    ):
        assert f">{column}</th>" in endpoint_table
    # `drift` and `errors` are constant across this fixture's three specs, so the HTML page
    # drops them both (item 6); the shared text table still carries them, which is what
    # `test_report.py` checks.
    assert ">drift</th>" not in endpoint_table
    assert ">errors</th>" not in endpoint_table
    assert "<td>@novita</td>" in endpoint_table and "<td>@gmicloud</td>" in endpoint_table
    for column in ("turn", "prompt", "hits", "ttl"):
        assert f">{column}</th>" in rung_table
    # `cached cold` and `errors` are constant `0` across this fixture's rungs, so the HTML
    # page drops them too.
    assert ">cached cold</th>" not in rung_table
    assert ">errors</th>" not in rung_table
    assert "<td>20,410</td>" not in endpoint_table  # the numbers stay as the terminal formats them
    assert "<td>20410</td>" in rung_table


def test_price_columns_stay_in_in_cache_eff_order_with_this_trace_right_after_eff() -> None:
    html = render_html(probe_document())
    endpoint_table = html.split('<div class="scroll">')[1]
    header = endpoint_table[: endpoint_table.index("</thead>")]
    assert re.search(
        r">in \$/M</th>.*>cache \$/M</th>.*>eff \$/M</th>.*>this trace \$</th>", header
    )
    row = endpoint_table[endpoint_table.index("<td>@novita</td>") :]
    # in $/M, cache $/M, eff $/M, this trace $ -- in that order, right after each other
    assert re.search(r"<td>0\.300</td><td>0\.030</td><td>0\.041</td><td>\$0\.0021</td>", row)


def test_endpoint_table_has_a_group_header_row_over_cache_price_and_speed() -> None:
    """Item 3: a colspan row above the column header names the cache/price/speed groups;
    `endpoint`, `errors` and `drift` are not grouped."""
    html = render_html(probe_document())
    endpoint_table = html.split('<div class="scroll">')[1]
    groups = endpoint_table[: endpoint_table.index("</tr>")]
    assert '<tr class="groups"><th></th>' in groups
    assert '<th colspan="3" class="group-start">cache</th>' in groups
    assert '<th colspan="4" class="group-start">price</th>' in groups
    assert '<th colspan="4" class="group-start">speed</th>' in groups


def test_method_sentence_states_turn_count_sizes_and_repeat_range() -> None:
    """Item 4: the method sentence names how many turns, how big and how many repeats, built
    from this run's own options and rung sizes rather than a fixed template."""
    html = render_html(probe_document())
    assert (
        '<p class="method">The run replayed 2 turns of a recorded coding session (20k and '
        "41k prompt tokens) against each endpoint: one uncached request to write the cache, "
        "then 1 repeat request to read it.</p>" in html
    )


def test_endpoint_table_caption_is_static_and_there_is_no_answer_paragraph() -> None:
    """Item 1: the old one-sentence verdict is gone; the caption is a fact about the table
    that survives however the run's own numbers turn out, not a claim about one row."""
    html = render_html(probe_document())
    assert f'<p class="caption">{ENDPOINT_CAPTION_HTML}' in html
    assert '<p class="answer"' not in html


def test_endpoint_table_caption_does_not_change_when_every_endpoint_errored() -> None:
    document = probe_document()
    entries = [entry for entry in map(as_document, as_list(document["summaries"]) or []) if entry]
    document["summaries"] = [{**entry, "eff_per_m_prompt": None, "errors": 1} for entry in entries]
    html = render_html(document)
    assert f'<p class="caption">{ENDPOINT_CAPTION_HTML}' in html
    assert '<p class="answer"' not in html


def test_this_trace_column_reports_what_a_session_like_the_trace_would_bill() -> None:
    html = render_html(probe_document())
    endpoint_table = html.split('<div class="scroll">')[1]
    assert ">this trace $</th>" in endpoint_table
    assert "<td>$0.0021</td>" in endpoint_table  # @novita: eff $/M x the trace's own tokens
    assert "<td>$0.0150</td>" in endpoint_table  # @gmicloud
    assert "<td>$0.0015</td>" in endpoint_table  # evil<spec>:model-z
    # item 12.5: the run's own spend is a caveat sentence now, not a `spent $X` table caption.
    assert (
        "This run cost $0.1776 in API spend; the estimate before running, assuming no cache "
        "hit, was $0.4851." in html
    )
    assert '<p class="caption">spent ' not in html


def test_dropped_columns_are_silently_omitted_not_announced() -> None:
    """Item 12.7: the old `omitted: no values` line is gone; a constant column just isn't
    there, in either table, and nothing on the page says so."""
    html = render_html(probe_document())
    endpoint_table, rung_table = html.split('<div class="scroll">')[1:3]
    assert ">errors</th>" not in endpoint_table
    assert ">drift</th>" not in endpoint_table
    assert ">cached cold</th>" not in rung_table
    assert ">errors</th>" not in rung_table
    assert "omitted" not in html


def test_html_report_reads_legacy_ttl_offset_field() -> None:
    document = probe_document()
    summaries = as_list(document["summaries"]) or []
    summary = as_document(summaries[0])
    assert summary is not None
    rungs = as_list(summary["rungs"]) or []
    rung = as_document(rungs[0])
    assert rung is not None
    ttl_reads = as_list(rung["ttl"]) or []
    ttl = as_document(ttl_reads[0])
    assert ttl is not None
    ttl["offset"] = ttl.pop("offset_s")

    html = render_html(document)

    assert "60s:1" in html


def test_drift_cell_is_empty_when_clean_and_a_provider_marker_reads_served_by() -> None:
    """Item 5: `-` means "not measured"; a clean row's drift cell is simply empty, and the
    `provider` marker is spelled out as `served by <name>` rather than left as a category."""
    document = probe_document()
    entries = [entry for entry in map(as_document, as_list(document["summaries"]) or []) if entry]
    document["summaries"] = [
        {**entries[0], "drift": "provider", "served": "SiliconFlow"},
        *entries[1:],
    ]
    html = render_html(document)
    endpoint_table = html.split('<div class="scroll">')[1]
    assert ">drift</th>" in endpoint_table  # no longer constant once one row drifts
    assert "<td>served by SiliconFlow</td>" in endpoint_table
    assert "<td></td>" in endpoint_table  # the two clean rows' cells are empty, not `-`


def test_drift_cell_names_another_provider_when_served_is_missing() -> None:
    """A `provider` marker with no served name still names something a reader can act on,
    rather than an empty cell that reads as clean. `served` defaults to `-`, which
    `_drift_cell` treats the same as missing, so leaving the field out reaches that branch."""
    document = probe_document()
    entries = [entry for entry in map(as_document, as_list(document["summaries"]) or []) if entry]
    first = {key: value for key, value in entries[0].items() if key != "served"}
    document["summaries"] = [{**first, "drift": "provider"}, *entries[1:]]
    html = render_html(document)
    endpoint_table = html.split('<div class="scroll">')[1]
    assert "<td>served by another provider</td>" in endpoint_table


def test_hits_cell_counts_what_hit_percent_counts() -> None:
    """Item 7: the `hits` cell's `n/total hit` counts any cached share as a hit, the same
    count `hit %` and the cache-share chart already use, with the reads spelled out on hover."""

    def _summary(label: str, records: list[dict[str, object]]) -> Document:
        summary = summarize_probe(label, [ProbeResult.model_validate(r) for r in records])
        return probe_summary_to_document(summary)

    six_hit_one_partial = [
        probe_record(spec_label="a", prompt_total=1_000),
        *[
            probe_record(
                spec_label="a", role="warm", attempt=i, cached=1_000, prompt_total=1_000, seq=i + 1
            )
            for i in range(1, 6)
        ],
        probe_record(spec_label="a", role="warm", attempt=6, cached=500, prompt_total=1_000, seq=7),
    ]
    four_hit_two_failed = [
        probe_record(spec_label="b", prompt_total=1_000),
        *[
            probe_record(
                spec_label="b", role="warm", attempt=i, cached=1_000, prompt_total=1_000, seq=i + 1
            )
            for i in range(1, 4)
        ],
        probe_record(
            spec_label="b", role="warm", attempt=4, cached=1_000, prompt_total=1_000, seq=5
        ),
        probe_record(
            spec_label="b", role="warm", attempt=5, status=502, error="bad gateway", seq=6
        ),
        probe_record(
            spec_label="b", role="warm", attempt=6, status=502, error="bad gateway", seq=7
        ),
    ]
    document: Document = {
        "run_dir": "/runs/t/20260101-000000",
        "trace": "t",
        "conversation": "c1",
        "created": "20260101-000000",
        "run_hex": None,
        "protocol": "probe",
        "options": {
            "rungs": [1],
            "repeats": [6],
            "gap_s": 1.0,
            "warm": False,
            "throughput": False,
            "ttl_s": None,
        },
        "summaries": [_summary("a", six_hit_one_partial), _summary("b", four_hit_two_failed)],
        "output_file": None,
        "changed": False,
    }
    html = render_html(document)
    assert (
        '<td title="hit · hit · hit · hit · hit · hit, 50% cached">6/6 hit, 1 partial</td>' in html
    )
    assert '<td title="hit · hit · hit · hit · failed · failed">4/6 hit, 2 failed</td>' in html


def test_the_page_follows_the_new_reading_order() -> None:
    """Item 11: header, method sentence, endpoint table, the two charts, then the per-turn
    table closed inside a `<details>`, then the caveats and the footer."""
    html = render_html(probe_document())
    assert html.count('<section class="table">') == 2
    header = html.index("<header")
    method = html.index('<p class="method">')
    provider = html.index("<h3>Per provider</h3>")
    caption = html.index(f'<p class="caption">{ENDPOINT_CAPTION_HTML}')
    cache_chart = html.index("<h3>Share of prompt tokens served from cache, per turn</h3>")
    cost_chart = html.index("<h3>What a session like this one bills</h3>")
    details = html.index('<details class="turns"><summary>Per turn: every request</summary>')
    caveats = html.index('<section class="caveats">')
    footer = html.index('<footer class="foot">')
    assert (
        header < method < provider < caption < cache_chart < cost_chart < details < caveats < footer
    )
    assert "<h3>Per rung</h3>" not in html
    assert "<h2>Cache and cost</h2>" not in html
    assert html.index('class="scroll"', provider) < details  # the endpoint table under its own head


def test_charts_scale_to_the_page_instead_of_clipping() -> None:
    html = render_html(probe_document())
    desktop_css, phone_css = html.split("@media (max-width: 900px)")
    assert "min-width" not in desktop_css  # the chart scales with the column on a desktop
    assert "min-width: 640px" in phone_css  # and scrolls at a readable size on a phone
    for svg in _SVG.findall(html):
        root = svg[: svg.index(">") + 1]
        assert 'viewBox="0 0 ' in root
        assert 'width="' not in root and 'height="' not in root


def test_interpolated_values_are_escaped() -> None:
    html = render_html(probe_document())
    assert "evil&lt;spec&gt;:model-z" in html
    assert "<spec>" not in html
    assert "&amp;" not in html  # nothing to escape yet, and no double-escaped text either


def test_dark_mode_is_a_media_query_over_the_same_slots() -> None:
    html = render_html(probe_document())
    assert "@media (prefers-color-scheme: dark)" in html
    light = html.split("@media (prefers-color-scheme: dark)")[0]
    dark = html.split("@media (prefers-color-scheme: dark)")[1]
    assert "--surface: #fcfcfb" in light and "--surface: #1a1a19" in dark
    assert "--series-1: #2a78d6" in light and "--series-1: #3987e5" in dark
    assert "--series-3: #1baf7a" in light and "--series-3: #199e70" in dark


# --- the probe charts ---------------------------------------------------------


def test_hit_rate_chart_has_one_bar_per_spec_per_rung() -> None:
    html = render_html(probe_document())
    charts = _SVG.findall(html)
    assert len(charts) == 2
    # novita measured two rungs, gmicloud and the third spec one each
    assert _bars(charts[0]) == 4
    # the session-cost chart draws a light and a dark bar for every priced endpoint
    assert _bars(charts[1]) == 6


def test_rungs_share_one_x_label_and_the_different_sizes_do_not_split_them() -> None:
    """The third spec's 9k rung is the same rung: one group, one label, sized by the first."""
    html = render_html(probe_document())
    bars = _SVG.findall(html)[0]
    labels = re.findall(r'<text class="tick mid"[^>]*>([^<]+)</text>', bars)
    # item 8: "rung" becomes "turn" in the HTML page's own chart and column labels.
    assert labels == ["turn 1 · 20k", "turn 2 · 41k"]
    assert "9k" not in bars


def test_the_grouped_chart_names_both_axes() -> None:
    bars = _SVG.findall(render_html(probe_document()))[0]
    assert '<text class="axis-title" x="46.0" y="22.0">share %</text>' in bars
    assert '<text class="axis-title" x="1060.0" y="334.0">prompt size (turn)</text>' in bars
    assert bars.count('class="axis-title"') == 2


def test_a_clipped_row_label_keeps_its_full_text_on_hover() -> None:
    """`_row_label`, shared by every horizontal-bar chart; `horizontal_bars` itself is dead
    code (no caller since item 10's paired chart replaced the price bars) and was deleted, so
    this exercises the shared label-clipping through `paired_horizontal_bars` instead."""
    long_label = "openrouter:deepseek/deepseek-v4.1-flash@novita"
    bar = PairedBar(value=0.041, value_text="0.041", title="cost")
    row = PairedBarRow(series=0, label=long_label, light=bar, dark=bar)
    svg = paired_horizontal_bars([row], label="session bill per endpoint")
    assert f"<title>{long_label}</title>" in svg
    assert "…" in svg

    short = paired_horizontal_bars([replace(row, label="@novita")], label="session bill")
    # the chart's own title plus one per bar half (light and dark), none for a label that fits
    assert short.count("<title>") == 3


def test_hit_rate_chart_keeps_a_slot_per_spec_and_labels_every_bar() -> None:
    html = render_html(probe_document())
    bars = _SVG.findall(html)[0]
    assert 'class="bar s1"' in bars and 'class="bar s2"' in bars and 'class="bar s3"' in bars
    assert bars.count("<title>") == _bars(bars) + 1  # one per bar, plus the chart's own
    assert (
        "or:model@novita · rung 1 · 20,410 prompt tokens · "
        "hit 100.0% (1/1 warm reads) · cached 93.6%" in bars
    )
    assert "hit 0.0% (0/1 warm reads)" in bars
    assert '<text class="tick mid"' in bars


def test_session_cost_chart_pairs_a_listed_bar_and_a_measured_bar_per_endpoint() -> None:
    """Item 10: the price-bar chart is gone; each endpoint gets two bars, same series colour,
    one at the listed cache price for a session like the trace and one at what it measured."""
    html = render_html(probe_document())
    cost = _SVG.findall(html)[1]
    assert _bars(cost) == 6  # a light and a dark bar per endpoint
    assert 'class="bar light s1"' in cost and 'class="bar s1"' in cost
    assert "listed $0.030/M cache read" in cost
    assert "eff $0.041/M prompt, hit-weighted h 95.8%" in cost
    assert "price source: OpenRouter listing" in cost
    assert '<text class="row-label" x="0.0"' in cost
    # the listed bar and the measured bar both show a session-shaped dollar amount, not $/M
    assert ">$0.0015</text>" in cost and ">$0.0021</text>" in cost and ">$0.0150</text>" in cost
    assert "/M<" not in cost  # no $/M label leaks into a value node, only into the titles


def test_session_cost_chart_has_its_own_legend_and_caption() -> None:
    html = render_html(probe_document())
    assert (
        '<ul class="legend"><li><span class="swatch shade-light"></span>at the listed cache '
        'price, every repeat a hit</li><li><span class="swatch shade-dark"></span>at the '
        "measured hit rate</li></ul>" in html
    )
    assert (
        "Light: what the price list promises when the cache always hits. Dark: what this run "
        "measured. The difference is what the misses cost." in html
    )


def test_prices_show_three_decimals_in_the_endpoint_table() -> None:
    html = render_html(probe_document())
    endpoint_table = html.split('<div class="scroll">')[1]
    assert "<td>0.041</td>" in endpoint_table  # eff $/M, not 0.0 or 0.04
    assert "<td>0.300</td>" in endpoint_table  # in $/M
    assert "<td>0.030</td>" in endpoint_table  # cache $/M


def test_a_spec_without_a_price_gets_a_dash_row_in_the_session_cost_chart() -> None:
    """Item 10: a row with either half unknown draws no bars at all, never a lone bar next
    to an empty slot that would read as a zero it never measured."""
    document = probe_document()
    entries = [entry for entry in map(as_document, as_list(document["summaries"]) or []) if entry]
    document["summaries"] = [
        {**entries[0], "cache_read_price": None, "session_prompt_usd": None},
        *entries[1:],
    ]
    cost = _SVG.findall(render_html(document))[1]
    assert _bars(cost) == 4  # the two fully priced specs keep both their bars
    assert cost.count('class="row-label"') == 3  # every spec keeps its label
    assert '<text class="value"' in cost and ">-</text>" in cost


def test_specs_past_the_eight_slots_stay_in_the_tables_and_are_named() -> None:
    document = probe_document()
    entries = [entry for entry in map(as_document, as_list(document["summaries"]) or []) if entry]
    [first] = entries[:1]
    document["summaries"] = [{**first, "label": f"or:model@r{index}"} for index in range(9)]
    html = render_html(document)
    charts = _SVG.findall(html)
    assert _bars(charts[0]) == MAX_SERIES * 2  # the fixture entry's two rungs, eight times
    assert _bars(charts[1]) == MAX_SERIES * 2  # a light and a dark bar for each charted spec
    assert "8 of 9 endpoints are charted; the rest are in the tables above: @r8" in html
    assert "<td>@r0</td>" in html and "<td>@r8</td>" in html


def test_empty_probe_document_still_renders_a_page() -> None:
    document = probe_document()
    document["summaries"] = []
    html = render_html(document)
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    assert "<svg" not in html


# --- the full replay ----------------------------------------------------------


def test_full_replay_renders_the_summary_table_and_one_curve_per_spec() -> None:
    html = render_html(replay_document())
    assert '<th scope="col">label</th>' in html
    assert '<th scope="col">eff $/M prompt</th>' in html
    assert "<td>deepseek:model-a</td>" in html
    curves = _SVG.findall(html)
    assert len(curves) == 2
    for curve in curves:
        assert curve.count('class="line"') == 1
        assert curve.count('class="marker"') == 4
    assert "cached" in curves[0] and "turn 1:" in curves[0]


def test_full_replay_curve_carries_the_cached_fraction_of_every_turn() -> None:
    html = render_html(replay_document())
    cold, warm = _SVG.findall(html)
    assert "turn 1: 0.0% of the prompt cached" in cold
    assert "turn 2: 50.0% of the prompt cached" in warm
    assert "turn 4: 100.0% of the prompt cached" in warm


# --- caveats and the error-reason note -----------------------------------------


def test_a_warm_runs_page_states_the_cache_mode_too() -> None:
    """The open item from round 1's review: a cold run's page says every hit was written by
    this run; a warm run is exactly the case where a hit may not have been, and the page must
    say so too, not just the `warm ms` column label."""
    document = probe_document()
    options = as_document(document["options"]) or {}
    document["options"] = {**options, "warm": True}
    html = render_html(document)
    # `escape()` renders the quotes as `&quot;`, the same as every other caveat sentence.
    sentence = (
        "&quot;warm&quot;: the cache was not reset between requests, so a hit in this table "
        "may have been written by earlier traffic, not by this run."
    )
    assert sentence in html
    assert html.count(sentence) == 1


def test_caveats_heading_and_plain_endpoint_label() -> None:
    html = render_html(failing_probe_document())
    assert "<h2>Caveats</h2>" in html
    # the fixed ordering (item 12) puts the nonce and blanks caveats ahead of this one, so
    # the anchor is no longer necessarily `cav-1`; the content is what a reader looks for.
    assert re.search(r'<li id="cav-\d+">@novita 1 warm read failed \(HTTP 502\)</li>', html)
    # every caveat names its endpoint the same way: inside the sentence, never bolded
    caveats = html[html.index('<section class="caveats">') :]
    assert "<strong>" not in caveats


def test_a_2xx_carrying_an_error_payload_is_not_named_after_its_status() -> None:
    """A gateway answers `200` and puts the upstream failure in the body, so the status is
    not the reason: `failed (HTTP 200)` would read as a success and a failure at once."""
    records = [
        probe_record(prompt_total=1_000),
        probe_record(role="warm", attempt=1, cached=1_000, prompt_total=1_000, seq=2),
        probe_record(role="stream", attempt=0, seq=3, error='{"message": "upstream error"}'),
    ]
    summary = summarize_probe(
        "or:model@novita", [ProbeResult.model_validate(record) for record in records]
    )
    assert "1 throughput request failed (provider error)" in summary.notes
    assert not any("HTTP 200" in note for note in summary.notes)


def test_a_failed_warm_read_gets_an_error_reason_note() -> None:
    """Item 8: the records' own `status`/`error` become a note naming what failed, so an
    `errors` count is never left unexplained next to the tables."""
    html = render_html(failing_probe_document())
    assert "1 warm read failed (HTTP 502)" in html


def test_burst_caveat_merges_endpoints_that_share_the_same_burst_turn() -> None:
    """Item 12.3: two endpoints whose streamed request burst on the same turn get one shared
    sentence naming both, instead of one line repeated per endpoint."""

    def _summary(label: str, records: list[dict[str, object]]) -> Document:
        summary = summarize_probe(label, [ProbeResult.model_validate(r) for r in records])
        return probe_summary_to_document(summary)

    novita = [
        probe_record(spec_label="or:model@novita", prompt_total=20_000),
        probe_record(
            spec_label="or:model@novita",
            role="warm",
            attempt=1,
            cached=20_000,
            prompt_total=20_000,
            seq=2,
        ),
        probe_record(
            spec_label="or:model@novita", role="stream", ttft_ms=500.0, fingerprint="tok", seq=3
        ),
    ]
    gmicloud = [
        probe_record(spec_label="or:model@gmicloud", provider="GMICloud", prompt_total=20_000),
        probe_record(
            spec_label="or:model@gmicloud",
            role="warm",
            attempt=1,
            cached=20_000,
            prompt_total=20_000,
            seq=2,
        ),
        probe_record(
            spec_label="or:model@gmicloud", role="stream", ttft_ms=600.0, fingerprint="tok", seq=3
        ),
    ]
    document: Document = {
        "run_dir": "/runs/t/20260101-000000",
        "trace": "t",
        "conversation": "c1",
        "created": "20260101-000000",
        "run_hex": None,
        "protocol": "probe",
        "options": {
            "rungs": [1],
            "repeats": [1],
            "gap_s": 1.0,
            "warm": False,
            "throughput": True,
            "ttl_s": None,
        },
        "summaries": [_summary("or:model@novita", novita), _summary("or:model@gmicloud", gmicloud)],
        "output_file": None,
        "changed": False,
    }
    html = render_html(document)
    # `escape()` renders the apostrophe as `&#x27;`, the same as every other caveat sentence.
    assert (
        "@novita and @gmicloud delivered the 20k-token turn&#x27;s answer in one burst, so "
        "tok/s for that turn is -; their tok/s in the summary is the median of the other "
        "turns." in html
    )
    assert html.count("delivered the 20k-token turn&#x27;s answer in one burst") == 1


def test_run_cost_sentence_names_the_precheck_share_when_there_was_one() -> None:
    """Item 12.5: the same fields `spend_lines` reads, as one caveat sentence; the pre-check
    share is only mentioned when the run actually spent something checking availability."""
    without = render_html(probe_document())
    assert (
        "This run cost $0.1776 in API spend; the estimate before running, assuming no cache "
        "hit, was $0.4851." in without
    )
    assert "availability check" not in without

    with_precheck = probe_document()
    with_precheck["precheck"] = {"requests": 3, "spend_usd": 0.0012, "worst_case_usd": 0.003}
    html = render_html(with_precheck)
    assert (
        "This run cost $0.1776 in API spend ($0.0012 of it on the availability check); the "
        "estimate before running, assuming no cache hit, was $0.4851." in html
    )


def test_run_cost_sentence_ends_plainly_without_a_worst_case() -> None:
    """A run whose worst case is unknown gets a full stop instead of the estimate clause,
    rather than a sentence with a hole where the number would go."""
    document = probe_document()
    del document["worst_case_usd"]
    html = render_html(document)
    assert "This run cost $0.1776 in API spend." in html
    assert "estimate before running" not in html


@pytest.mark.parametrize("document", [probe_document(), replay_document()])
def test_every_document_carries_its_own_styles_and_nothing_else(document: Document) -> None:
    html = render_html(document)
    assert html.count("<style>") == 1
    assert "@import" not in html and "src=" not in html and "url(" not in html
