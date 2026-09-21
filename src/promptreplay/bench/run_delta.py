"""Two runs subtracted spec by spec: what each metric was, and how much it moved.

`bench/run_history` reads a runs directory as a time series; this is the other half of the
time axis, the arithmetic of `compare`. Every metric is a pair — the value in the earlier
run and in the later one — and the delta between them, which is `None` whenever either
side has no number to subtract. That is deliberate: a spec that measured nothing in one of
the two runs did not measure zero, and printing a delta against zero would invent a
collapse the run never reported.

The rows come out in the order the earlier run measured them. A sweep orders its own rows
by the effective price it measured, and that order is the verdict a comparison is read
against: a spec whose price rose is spotted by where it sat in the first run, not by an
alphabetical column.

OpenRouter renames a provider's own tag within hours, not weeks (`gmicloud` becomes
`gmicloud/fp8` the same day, in practice), so pairing by label alone turns a routine rename
into two unrelated rows in `only_in_a`/`only_in_b`. After the label pairing, the rows left
over on each side are paired again by provider name -- `RunRow.provider` before the first
`/`, the part a reader actually compares across runs, on the same target and model, since
the tag is the only part of a label a rename touches -- but only when that leaves exactly
one candidate on each side; two rows sharing a provider name on either side is ambiguous
(which one replaced which?), so both stay unpaired rather than guessing. A row still pairs
at most once.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from promptreplay.bench.run_history import RunNumbers, RunRow

__all__ = ["Comparison", "MetricPair", "PriceDelta", "SpecDelta", "compare_runs"]


class MetricPair(BaseModel):
    """One metric in both runs of a comparison, and the change between them."""

    a: float | None = None
    b: float | None = None

    @property
    def delta(self) -> float | None:
        """`b - a`, or `None` when either side has no number to subtract."""
        if self.a is None or self.b is None:
            return None
        return self.b - self.a


class SpecDelta(BaseModel):
    """One spec present in both runs: its numbers there and here."""

    spec: str
    paired_with: str | None = None
    """The later run's label, when a provider-name fallback paired it with a different one
    than `spec`; `None` for an ordinary label pair, where the two are the same string."""
    hit_rate: MetricPair = Field(default_factory=MetricPair)
    eff_per_m_prompt: MetricPair = Field(default_factory=MetricPair)
    ttft_ms: MetricPair = Field(default_factory=MetricPair)
    gen_tok_s: MetricPair = Field(default_factory=MetricPair)


class PriceDelta(BaseModel):
    """One spec both runs recorded a listed price for: the two prompt-side prices.

    A cache benchmark is about the cache-read price as much as the input one — a provider
    that raises its cache-read rate changes what a warm run costs — so both are carried.
    """

    spec: str
    paired_with: str | None = None
    """See `SpecDelta.paired_with`."""
    price_in: MetricPair = Field(default_factory=MetricPair)
    price_cache_read: MetricPair = Field(default_factory=MetricPair)


class Comparison(BaseModel):
    """Two runs of one trace and protocol, and every spec either of them measured."""

    a: RunNumbers
    b: RunNumbers
    rows: list[SpecDelta] = Field(default_factory=list[SpecDelta])
    listed: list[PriceDelta] = Field(default_factory=list[PriceDelta])
    only_in_a: list[str] = Field(default_factory=list[str])
    only_in_b: list[str] = Field(default_factory=list[str])


def compare_runs(a: RunNumbers, b: RunNumbers) -> Comparison:
    """The delta per spec of two runs, in the order the first of them measured them.

    A spec only one of the runs measured cannot be subtracted at all, so it is named apart
    instead of paired, and so is a spec only one of the runs recorded a price for -- unless
    the provider-name fallback above pairs it with a differently labelled row instead.
    """
    here = {row.spec: row for row in b.rows}
    label_matches = {row.spec: here[row.spec] for row in a.rows if row.spec in here}
    unpaired_a = [row for row in a.rows if row.spec not in label_matches]
    unpaired_b = [row for row in b.rows if row.spec not in label_matches]
    fallback_matches = _fallback_matches(unpaired_a, unpaired_b)
    pairs = [
        (row, later)
        for row in a.rows
        if (later := label_matches.get(row.spec) or fallback_matches.get(row.spec)) is not None
    ]
    matched_b_specs = {later.spec for _, later in pairs}
    return Comparison(
        a=a,
        b=b,
        rows=[_delta(earlier, later) for earlier, later in pairs],
        listed=[
            PriceDelta(
                spec=earlier.spec,
                paired_with=_paired_with(earlier, later),
                price_in=MetricPair(a=earlier.listed_input, b=later.listed_input),
                price_cache_read=MetricPair(a=earlier.listed_cache_read, b=later.listed_cache_read),
            )
            for earlier, later in pairs
            if earlier.listed_input is not None and later.listed_input is not None
        ],
        only_in_a=[row.spec for row in a.rows if row.spec not in {p[0].spec for p in pairs}],
        only_in_b=[row.spec for row in b.rows if row.spec not in matched_b_specs],
    )


def _fallback_matches(
    unpaired_a: Sequence[RunRow], unpaired_b: Sequence[RunRow]
) -> dict[str, RunRow]:
    """Provider-name pairing over the rows the label pass left over (see the module docstring).

    Keyed by the earlier row's `spec`, so `compare_runs` can look a match up the same way it
    looks up a label match.
    """
    a_by_provider: dict[tuple[str, str, str], list[RunRow]] = {}
    for row in unpaired_a:
        a_by_provider.setdefault(_provider_key(row), []).append(row)
    b_by_provider: dict[tuple[str, str, str], list[RunRow]] = {}
    for row in unpaired_b:
        b_by_provider.setdefault(_provider_key(row), []).append(row)
    matches: dict[str, RunRow] = {}
    for key, a_rows in a_by_provider.items():
        b_rows = b_by_provider.get(key, [])
        if len(a_rows) == 1 and len(b_rows) == 1:
            matches[a_rows[0].spec] = b_rows[0]
    return matches


def _provider_key(row: RunRow) -> tuple[str, str, str]:
    """What a renamed tag keeps: the same target and model, and the provider its tag names
    with any variant/quantization suffix dropped.

    The target and the model are part of the key because only the tag is what OpenRouter
    renames. Without them two runs of one trace on different models -- which `compare`
    allows, it only refuses a different trace or protocol -- would pair every provider they
    have in common and call it a rename.
    """
    return row.target, row.model, row.provider.partition("/")[0]


def _paired_with(earlier: RunRow, later: RunRow) -> str | None:
    return later.spec if later.spec != earlier.spec else None


def _delta(earlier: RunRow, later: RunRow) -> SpecDelta:
    return SpecDelta(
        spec=earlier.spec,
        paired_with=_paired_with(earlier, later),
        hit_rate=MetricPair(a=earlier.hit_rate, b=later.hit_rate),
        eff_per_m_prompt=MetricPair(a=earlier.eff_per_m_prompt, b=later.eff_per_m_prompt),
        ttft_ms=MetricPair(a=earlier.ttft_ms, b=later.ttft_ms),
        gen_tok_s=MetricPair(a=earlier.gen_tok_s, b=later.gen_tok_s),
    )
