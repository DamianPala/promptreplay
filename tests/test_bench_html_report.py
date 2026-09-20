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
from provibench.bench.html_svg import MAX_SERIES, BarRow, horizontal_bars
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
            label, [ProbeResult.model_validate(record) for record in entries], prices=prices[label]
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


def test_page_carries_the_run_and_its_cache_mode() -> None:
    html = render_html(probe_document())
    header = html[html.index("<header") : html.index("</header>")]
    assert "<h1>t</h1>" in header
    # the three specs name two models between them, in spec order and without repeats
    assert '<p class="sub">model, model-z · cold (nonce)' in header
    assert "2026-01-01 00:00:00 · run abc123def456</p>" in header
    # the fixture's run_dir is absolute; the page shows the run, never the operator's paths
    assert '<p class="where">runs/t/20260101-000000</p>' in header
    assert "/runs/" not in html


def test_both_tables_keep_the_terminal_columns_and_short_labels() -> None:
    html = render_html(probe_document())
    assert '<p class="caption">endpoints: or:model@&lt;provider&gt;</p>' in html
    endpoint_table, rung_table = html.split('<div class="scroll">')[1:3]
    assert '<th scope="col">endpoint</th>' in endpoint_table
    for column in ("hit %", "1st hit %", "cached %", "eff $/M", "TTFT ms", "tok/s", "drift"):
        assert f'<th scope="col">{column}</th>' in endpoint_table
    assert "<td>@novita</td>" in endpoint_table and "<td>@gmicloud</td>" in endpoint_table
    for column in ("rung", "prompt", "cached cold", "hits", "ttl"):
        assert f'<th scope="col">{column}</th>' in rung_table
    assert "<td>20,410</td>" not in endpoint_table  # the numbers stay as the terminal formats them
    assert "<td>20410</td>" in rung_table


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


def test_each_probe_table_carries_its_own_heading() -> None:
    html = render_html(probe_document())
    assert html.count('<section class="table">') == 2
    provider = html.index("<h3>Per provider</h3>")
    rung = html.index("<h3>Per rung</h3>")
    assert provider < rung
    assert html.index('class="scroll"', provider) < rung  # the first table sits under its own head


def test_charts_scale_to_the_page_instead_of_clipping() -> None:
    html = render_html(probe_document())
    desktop_css, phone_css = html.split("@media (max-width: 640px)")
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
    assert _bars(charts[1]) == 3


def test_rungs_share_one_x_label_and_the_different_sizes_do_not_split_them() -> None:
    """The third spec's 9k rung is the same rung: one group, one label, sized by the first."""
    html = render_html(probe_document())
    bars = _SVG.findall(html)[0]
    labels = re.findall(r'<text class="tick mid"[^>]*>([^<]+)</text>', bars)
    assert labels == ["rung 1 · 20k", "rung 2 · 41k"]
    assert "9k" not in bars


def test_the_grouped_chart_names_both_axes() -> None:
    bars = _SVG.findall(render_html(probe_document()))[0]
    assert '<text class="axis-title" x="46.0" y="22.0">hit %</text>' in bars
    assert '<text class="axis-title" x="1060.0" y="334.0">rung</text>' in bars
    assert bars.count('class="axis-title"') == 2


def test_a_clipped_row_label_keeps_its_full_text_on_hover() -> None:
    long_label = "openrouter:deepseek/deepseek-v4.1-flash@novita"
    row = BarRow(series=0, label=long_label, value=0.041, value_text="0.041", title="cost")
    svg = horizontal_bars([row], label="effective prompt price per spec")
    assert f"<title>{long_label}</title>" in svg
    assert "…" in svg

    short = horizontal_bars([replace(row, label="@novita")], label="effective prompt price")
    # the chart's own title and the bar's, none for a label that fits
    assert short.count("<title>") == 2


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


def test_cost_chart_is_horizontal_bars_in_the_same_series_order() -> None:
    html = render_html(probe_document())
    cost = _SVG.findall(html)[1]
    assert _bars(cost) == 3
    assert "eff $0.041/M prompt" in cost
    assert "listed $0.30/M in" in cost
    assert "hit-weighted h 95.8%" in cost
    assert '<text class="row-label" x="0.0"' in cost


def test_a_spec_without_a_price_keeps_its_row_rather_than_vanishing() -> None:
    """An unknown price reads as `-` next to the other specs, never as a missing bar."""
    document = probe_document()
    entries = [entry for entry in map(as_document, as_list(document["summaries"]) or []) if entry]
    document["summaries"] = [{**entries[0], "eff_per_m_prompt": None}, *entries[1:]]
    cost = _SVG.findall(render_html(document))[1]
    assert _bars(cost) == 2  # the priced specs
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
    assert _bars(charts[1]) == MAX_SERIES
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


@pytest.mark.parametrize("document", [probe_document(), replay_document()])
def test_every_document_carries_its_own_styles_and_nothing_else(document: Document) -> None:
    html = render_html(document)
    assert html.count("<style>") == 1
    assert "@import" not in html and "src=" not in html and "url(" not in html
