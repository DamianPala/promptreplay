"""Drift: how each spec's measurement differs from the reference spec's.

Two providers of one model should return the same prompt tokenization and the same first
output tokens; when they do not, the endpoint is quantized, substituted, or templated
differently, and that is worth more to a reader than the number it sits next to. `drift` is
the short column that says so, and these are the comparisons behind it.

The reference is the first spec served natively — a native endpoint is the model as its
owner serves it, and therefore the tokenizer and the fingerprint the others are judged
against — or the first spec when every one of them is an OpenRouter endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from provibench.bench.openrouter import normalize_provider
from provibench.bench.probe_summary import ProbeSummary, RungSummary

DRIFT_TOKENS_PCT = 1.0
"""A prompt-size difference smaller than this is tokenizer noise, not drift."""

_FIELD_SEPARATOR = ","


class SpecRef(Protocol):
    """What `apply_drift`/`reported_average.apply_reported_average` need of a spec: its
    target's kind, the model it asked for, and the providers it pinned."""

    @property
    def kind(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def providers(self) -> list[str]: ...


def apply_drift(
    summaries: Sequence[ProbeSummary], specs: Sequence[SpecRef] | None = None
) -> list[ProbeSummary]:
    """Mark one summary as the reference and fill every `drift` marker against it."""
    if not summaries:
        return []
    kinds = [spec.kind for spec in specs] if specs is not None else []
    pinned = [list(spec.providers) for spec in specs] if specs is not None else []
    reference = next((index for index, kind in enumerate(kinds) if kind != "openrouter"), 0)
    ref_requested = _requested(pinned, reference)
    ref_kind = _kind(kinds, reference)
    ref_rung = largest(summaries[reference])
    ref_models = models_of(summaries[reference])
    for index, summary in enumerate(summaries):
        rung = largest(summary)
        incomplete = index != reference and _incomplete(rung, ref_rung)
        summary.tokens_delta_pct = None if incomplete else tokens_delta(rung, ref_rung)
        summary.fingerprint_match = fingerprint_match(rung, ref_rung)
        summary.reference = index == reference
        if incomplete:
            summary.notes = [*summary.notes, "token drift n/a (rung skipped)"]
        summary.drift = (
            None
            if index == reference
            else _markers(
                summary,
                _requested(pinned, index) or ref_requested,
                ref_models,
                same_kind=_kind(kinds, index) == ref_kind,
            )
        )
    return list(summaries)


def models_of(summary: ProbeSummary) -> list[str]:
    """The models one summary's responses named, in first-seen order."""
    return list(summary.models_seen)


def largest(summary: ProbeSummary) -> RungSummary | None:
    """The rung with the largest cold prompt: the size the token counts are compared at."""
    rungs = [rung for rung in summary.rungs if not rung.skipped and rung.prompt_cold > 0]
    return max(rungs, key=lambda rung: rung.prompt_cold, default=None)


def _incomplete(rung: RungSummary | None, reference: RungSummary | None) -> bool:
    """Whether the compared rungs are different turns: a size gap that is not tokenizer drift.

    Every spec in a run shares the same planned rung set, so a spec whose largest usable
    rung is not the reference's largest one is missing a rung the reference has -- skipped by
    a failed cold write, most often. Comparing prompt sizes across two different turns would
    read as `tokens±N%` drift that is really just two prompts of different sizes.
    """
    if rung is None or reference is None:
        return False
    return rung.rung != reference.rung


def tokens_delta(rung: RungSummary | None, reference: RungSummary | None) -> float | None:
    """The largest rung's prompt size against the reference's; `None` for the reference."""
    if rung is None or reference is None or reference.endpoint == rung.endpoint:
        return None
    if reference.prompt_cold <= 0:
        return None
    return (rung.prompt_cold - reference.prompt_cold) / reference.prompt_cold * 100


def fingerprint_match(rung: RungSummary | None, reference: RungSummary | None) -> bool | None:
    """Whether the two largest rungs fingerprint the same; `None` when either has none.

    Reported on the summary and left out of the drift markers: a mismatch is expected
    between quantizations, so it is an observation to read, not a warning to act on.
    """
    if rung is None or reference is None or rung.fingerprint is None:
        return None
    if reference.fingerprint is None or reference.endpoint == rung.endpoint:
        return None
    return rung.fingerprint == reference.fingerprint


def _requested(pinned: Sequence[list[str]], index: int) -> list[str]:
    """The providers a spec pinned; an unpinned spec is compared against the reference's."""
    return list(pinned[index]) if index < len(pinned) else []


def _kind(kinds: Sequence[str], index: int) -> str | None:
    """A spec's target kind, or `None` when the caller gave no specs to read it from."""
    return kinds[index] if index < len(kinds) else None


def _markers(
    summary: ProbeSummary,
    requested: Sequence[str],
    ref_models: Sequence[str],
    *,
    same_kind: bool,
) -> str | None:
    """The short drift markers for one spec: provider, model, tokens.

    The fingerprint is deliberately not one of them. At temperature 0 two providers serving
    the same weights still open with different sentences — quantization, a different chat
    template, a kernel's rounding — so a mismatch is the normal case, not a warning. The
    text is recorded for a reader to eyeball, and `fingerprint_match` says whether it agreed,
    but only the markers here are claims that something is wrong.
    """
    markers: list[str] = []
    if _provider_drift(summary, requested):
        markers.append("provider")
    if _model_drift(models_of(summary), ref_models, same_kind=same_kind):
        markers.append("model")
    delta = summary.tokens_delta_pct
    if delta is not None and abs(delta) >= DRIFT_TOKENS_PCT:
        markers.append(f"tokens{delta:+.0f}%")
    return _FIELD_SEPARATOR.join(markers) or None


def _provider_drift(summary: ProbeSummary, requested: Sequence[str]) -> bool:
    """Served differs from what was requested, or the spec was served by more than one.

    A pin is an endpoint tag, `relace/fp4`: the provider slug and, after the slash, the
    variant it serves. A response names only the provider (`Relace`), so the comparison is
    on the slug alone; the variant is not something a response can confirm or deny.
    """
    served = sorted(summary.providers_seen)
    if len(served) > 1:
        return True
    if not requested or not served:
        return False
    wanted = {normalize_provider(name.partition("/")[0]) for name in requested}
    return all(normalize_provider(name) not in wanted for name in served)


def _model_drift(seen: Sequence[str], ref_seen: Sequence[str], *, same_kind: bool) -> bool:
    """The responses named more than one model, or a model the reference did not name.

    A native endpoint and a gateway of the same weights answer with different `model`
    strings — `deepseek-flash` against `deepseek/deepseek-v4.1-flash` — so across kinds
    only a spec that varies within itself is a marker. Within one kind the strings are
    comparable, and a spec that names another model is the drift the column is for.
    """
    if len(seen) > 1:
        return True
    if not same_kind or not seen or not ref_seen:
        return False
    return seen[0] != ref_seen[0]
