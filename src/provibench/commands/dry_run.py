"""The `--dry-run` preview shared by `probe`, `sweep`, and `replay` (R4).

A dry run prices the same worst-case estimate a real run would and stops there: nothing is
sent, the confirmation never runs, `changed` is `false`, and `requires_confirmation` reports
what the same call without `--dry-run` and without `--yes` would do in a non-interactive
context (R4c) — independent both of whatever `--yes` this call was actually given, which
`--dry-run` accepts and ignores (R3c), and of whether this call's own streams could prompt.
A failure that does not depend on confirming, such as `--budget` refusing the estimate,
still applies to a dry run (R4b): only the confirmation gate itself is skipped.

The result document omits every field a real run can only produce by sending requests
(`run_dir`, `summaries`, and the like) and adds the estimate instead, per R4c's rule that the
shared schema declares such a field optional or nullable rather than predicting it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import click

from provibench.core.documents import (
    Document,
    JsonSchema,
    array,
    boolean,
    integer,
    nullable_number,
    nullable_object,
    nullable_string,
    number,
    obj,
    string,
)

if TYPE_CHECKING:
    from provibench.bench.estimate import PreCheckCost, SpecEstimate, UpperBound

DRY_RUN_OPTION = click.option(
    "--dry-run",
    is_flag=True,
    help="Price the run and stop: send nothing, exit 0, and print the estimate as the result",
)

_ESTIMATE_ITEM: JsonSchema = obj(
    {
        "label": string(),
        "tokens": integer(),
        "tokens_known": boolean(),
        "price": nullable_number(),
        "source": nullable_string(),
        "worst_case_usd": nullable_number(),
        "notes": array(string()),
    },
    required=["label", "tokens", "tokens_known", "price", "source", "worst_case_usd", "notes"],
)

_PRE_CHECK: JsonSchema = nullable_object(
    {"requests": integer(), "tokens": integer(), "usd": nullable_number()},
    required=["requests", "tokens", "usd"],
)

_UPPER_BOUND: JsonSchema = nullable_object(
    {"keep": integer(), "usd": number()}, required=["keep", "usd"]
)


def dry_run_fields(*, precheck: bool = False) -> dict[str, JsonSchema]:
    """The extra properties a `--dry-run` result adds to a probe, sweep, or replay schema.

    `precheck` adds `pre_check` and `upper_bound`, which only a `probe` or `sweep` availability
    check and `--top` ceiling can produce; `replay` has neither and omits both from its schema.
    """
    fields: dict[str, JsonSchema] = {
        "estimate": array(_ESTIMATE_ITEM),
        "total_usd": nullable_number(),
        "runs_dir": string(),
        "requires_confirmation": boolean(),
    }
    if precheck:
        fields["pre_check"] = _PRE_CHECK
        fields["upper_bound"] = _UPPER_BOUND
    return fields


def requires_confirmation() -> bool:
    """Whether the same call without `--dry-run` and without `--yes` would gate (R4c).

    R4c fixes the context of that question: it asks about a *non-interactive* context, not
    the one this call happens to run in, so the answer must not read the current streams.
    `probe`, `sweep`, and `replay` always reach a request that spends API credit — a spec
    list is never empty, since `probe`/`replay` require one on the command line and a sweep
    that selects nothing fails with `invalid_input` before this point — so the same call
    without `--yes` is always a *gated call* and the answer is always `True`.
    """
    return True


def build_document(
    estimates: Sequence[SpecEstimate],
    *,
    runs_dir: Path,
    pre_check: PreCheckCost | None = None,
    upper_bound: UpperBound | None = None,
) -> Document:
    """The `--dry-run` result document: the estimate, `partial: false`, `changed: false`."""
    from provibench.bench.estimate import estimate_total

    document: Document = {
        "estimate": [_estimate_item(estimate) for estimate in estimates],
        "total_usd": estimate_total(estimates, pre_check=pre_check),
        "runs_dir": str(runs_dir),
        "requires_confirmation": requires_confirmation(),
        "partial": False,
        "changed": False,
    }
    if pre_check is not None:
        document["pre_check"] = _pre_check_item(pre_check)
    if upper_bound is not None:
        document["upper_bound"] = {"keep": upper_bound.keep, "usd": upper_bound.usd}
    return document


def _estimate_item(estimate: SpecEstimate) -> Document:
    price = estimate.prices.prices.input if estimate.prices is not None else None
    source = estimate.prices.source if estimate.prices is not None else None
    return {
        "label": estimate.label,
        "tokens": estimate.tokens,
        "tokens_known": estimate.tokens_known,
        "price": price,
        "source": source,
        "worst_case_usd": estimate.usd,
        "notes": list(estimate.notes),
    }


def _pre_check_item(cost: PreCheckCost) -> Document:
    return {"requests": cost.requests, "tokens": cost.tokens, "usd": cost.usd}
