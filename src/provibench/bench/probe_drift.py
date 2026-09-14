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
    """What `apply_drift` needs of a spec: its target's kind, and the providers it pinned."""

    @property
    def kind(self) -> str: ...

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
    ref_rung = largest(summaries[reference])
    ref_models = models_of(summaries[reference])
    for index, summary in enumerate(summaries):
        rung = largest(summary)
        summary.tokens_delta_pct = tokens_delta(rung, ref_rung)
        summary.fingerprint_match = fingerprint_match(rung, ref_rung)
        summary.reference = index == reference
        summary.drift = (
            None
            if index == reference
            else _markers(summary, _requested(pinned, index) or ref_requested, ref_models)
        )
    return list(summaries)


def models_of(summary: ProbeSummary) -> list[str]:
    """The models one summary's responses named, in first-seen order."""
    return list(summary.models_seen)


def largest(summary: ProbeSummary) -> RungSummary | None:
    """The rung with the largest cold prompt: the size the token counts are compared at."""
    rungs = [rung for rung in summary.rungs if not rung.skipped and rung.prompt_cold > 0]
    return max(rungs, key=lambda rung: rung.prompt_cold, default=None)


def tokens_delta(rung: RungSummary | None, reference: RungSummary | None) -> float | None:
    """The largest rung's prompt size against the reference's; `None` for the reference."""
    if rung is None or reference is None or reference.spec == rung.spec:
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
    if reference.fingerprint is None or reference.spec == rung.spec:
        return None
    return rung.fingerprint == reference.fingerprint


def _requested(pinned: Sequence[list[str]], index: int) -> list[str]:
    """The providers a spec pinned; an unpinned spec is compared against the reference's."""
    return list(pinned[index]) if index < len(pinned) else []


def _markers(
    summary: ProbeSummary, requested: Sequence[str], ref_models: Sequence[str]
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
    if _model_drift(models_of(summary), ref_models):
        markers.append("model")
    delta = summary.tokens_delta_pct
    if delta is not None and abs(delta) >= DRIFT_TOKENS_PCT:
        markers.append(f"tokens{delta:+.0f}%")
    return _FIELD_SEPARATOR.join(markers) or None


def _provider_drift(summary: ProbeSummary, requested: Sequence[str]) -> bool:
    """Served differs from what was requested, or the spec was served by more than one."""
    served = sorted(summary.providers_seen)
    if len(served) > 1:
        return True
    if not requested or not served:
        return False
    wanted = {normalize_provider(name) for name in requested}
    return all(normalize_provider(name) not in wanted for name in served)


def _model_drift(seen: Sequence[str], ref_seen: Sequence[str]) -> bool:
    """The responses named more than one model, or a model the reference did not name."""
    if len(seen) > 1:
        return True
    if not seen or not ref_seen:
        return False
    return seen[0] != ref_seen[0]
