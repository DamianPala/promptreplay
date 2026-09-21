"""Tests for `bench.openrouter_stats`: parsing the two undocumented feed payloads.

Every payload here is either the checked-in fixtures (a real capture) or a small synthetic
dict for the edge cases the fixtures do not carry; nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from promptreplay.bench.openrouter_stats import canonical_slug, reported_average

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def hit_payload() -> dict[str, Any]:
    return _load("openrouter-stats-cache-hit.json")


@pytest.fixture
def pricing_payload() -> dict[str, Any]:
    return _load("openrouter-stats-effective-pricing.json")


@pytest.fixture
def models_payload() -> dict[str, Any]:
    return _load("openrouter-models-canonical.json")


_MODEL = "z-ai/glm-5.3-flash"
_PERMASLUG = "z-ai/glm-5.3-flash-20260826"
_DAY = "2026-09-20"


def _average(hit_payload: dict[str, Any], pricing_payload: dict[str, Any]) -> Any:
    return reported_average(
        hit_payload,
        pricing_payload,
        day=_DAY,
        model=_MODEL,
        permaslug=_PERMASLUG,
        fetched_at="2026-09-21T00:00:00+00:00",
    )


def test_reported_average_matches_the_fixtures_expected_shares(
    hit_payload: dict[str, Any], pricing_payload: dict[str, Any]
) -> None:
    average = _average(hit_payload, pricing_payload)
    assert average.day == _DAY
    assert average.model == _MODEL
    assert average.permaslug == _PERMASLUG
    assert average.shares["gmicloud"].share_pct == pytest.approx(69.6)
    assert average.shares["novita"].share_pct == pytest.approx(87.2)
    assert average.shares["streamlake"].share_pct == pytest.approx(84.4)
    assert average.shares["z-ai"].share_pct == pytest.approx(91.3)
    assert average.shares["sail-research"].share_pct == pytest.approx(92.7)


def test_reported_average_pools_two_sail_research_endpoints_by_tokens(
    hit_payload: dict[str, Any], pricing_payload: dict[str, Any]
) -> None:
    average = _average(hit_payload, pricing_payload)
    sail = average.shares["sail-research"]
    assert sail.endpoints == 2
    assert sail.tokens == 1872058719 + 1454792807


def test_reported_average_pools_by_tokens_not_by_endpoint_count(
    hit_payload: dict[str, Any], pricing_payload: dict[str, Any]
) -> None:
    """Baseten's two endpoints sit far enough apart, at weights far enough apart, that an
    unweighted mean would read 62.8: the fixture's `sail-research` pair alone cannot tell
    the two poolings apart."""
    average = _average(hit_payload, pricing_payload)
    assert average.shares["baseten"].share_pct == pytest.approx(63.5)


def test_reported_average_single_endpoint_provider_is_not_pooled(
    hit_payload: dict[str, Any], pricing_payload: dict[str, Any]
) -> None:
    average = _average(hit_payload, pricing_payload)
    gmicloud = average.shares["gmicloud"]
    assert gmicloud.endpoints == 1
    assert gmicloud.tokens == 11815690336


def test_reported_average_provider_absent_from_the_day_is_absent_from_shares() -> None:
    hit_payload = {"data": [{"x": "2026-09-20 00:00:00", "y": {"uuid-a": 80.0}}]}
    pricing_payload = {
        "data": {
            "endpointProviderSlugs": {"uuid-a": "alpha", "uuid-b": "beta"},
            "providerSummaries": [
                {"endpointId": "uuid-a", "totalTokens": 100},
                {"endpointId": "uuid-b", "totalTokens": 100},
            ],
        }
    }
    average = _average(hit_payload, pricing_payload)
    assert "alpha" in average.shares
    assert "beta" not in average.shares


def test_reported_average_missing_summary_falls_back_to_equal_weight() -> None:
    """One of a provider's two endpoints has no `providerSummaries` entry at all: it still
    pools, at the same weight as the one that does, instead of vanishing or scoring zero."""
    hit_payload = {"data": [{"x": "2026-09-20 00:00:00", "y": {"uuid-a": 80.0, "uuid-b": 40.0}}]}
    pricing_payload = {
        "data": {
            "endpointProviderSlugs": {"uuid-a": "alpha", "uuid-b": "alpha"},
            "providerSummaries": [{"endpointId": "uuid-a", "totalTokens": 1000}],
        }
    }
    average = _average(hit_payload, pricing_payload)
    alpha = average.shares["alpha"]
    assert alpha.endpoints == 2
    # uuid-a's real weight is not trusted alone against uuid-b's missing one: equal shares.
    assert alpha.share_pct == pytest.approx(60.0)
    assert alpha.tokens is None


def test_reported_average_raises_on_a_hit_payload_missing_data() -> None:
    with pytest.raises(ValueError, match="cache-hit-rate-comparison"):
        reported_average(
            {}, {"data": {}}, day=_DAY, model=_MODEL, permaslug=_PERMASLUG, fetched_at="x"
        )


def test_reported_average_raises_on_a_pricing_payload_missing_endpoint_slugs() -> None:
    hit_payload: dict[str, Any] = {"data": [{"x": "2026-09-20 00:00:00", "y": {}}]}
    with pytest.raises(ValueError, match="endpointProviderSlugs"):
        reported_average(
            hit_payload,
            {"data": {}},
            day=_DAY,
            model=_MODEL,
            permaslug=_PERMASLUG,
            fetched_at="x",
        )


def test_canonical_slug_found(models_payload: dict[str, Any]) -> None:
    assert canonical_slug(models_payload, "z-ai/glm-5.3-flash") == _PERMASLUG


def test_canonical_slug_not_found(models_payload: dict[str, Any]) -> None:
    assert canonical_slug(models_payload, "no/such-model") is None
