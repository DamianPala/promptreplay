"""The availability pre-check: one small request per candidate, after the confirmation.

An OpenRouter account can exclude endpoints — ignored providers, a data policy — and a
pinned request to one of them comes back as a 404 in well under a second ("0 endpoints out
of 1 requested are available matching your guardrail restrictions and data policy"). That is
cheap to learn, but only worth learning while the run is being planned: a candidate that
cannot be probed at all would otherwise take a `--top` slot and cost three failed colds the
report has to explain.

The pre-check sends the smallest rung's cold body once per candidate, `max_tokens 1` and no
nonce, so it costs the same as one warm read, and it goes through the probe's own request
path — the same headers, the same retries, the same one a cold request would use. A candidate
is kept when the request was served and removed with the API's own words when it was not.

Nothing here is sent before the caller's confirmation: the plan below is what the estimate
prices, and `precheck_candidates` is what runs once the run is agreed to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import httpx

from provibench.bench.probe_context import ProbeContext
from provibench.bench.probe_errors import error_type
from provibench.bench.probe_models import ProbeCall, ProbeOptions, ProbeResult, is_served
from provibench.bench.probe_requests import send
from provibench.bench.replay import prepare_body
from provibench.bench.requests import post
from provibench.bench.targets import RunSpec, resolve_api_key
from provibench.bench.trace import TraceEntry
from provibench.core.errors import InvalidInput

__all__ = ["CheckPlan", "PreCheckResult", "precheck_candidates", "unavailable_reason"]

_CONNECT_TIMEOUT_S = 30.0
_REASON_CAP = 80
_NOT_FOUND = 404
_PRIVACY_HINT = (
    " (change this at https://openrouter.ai/settings/privacy: allow this provider to train "
    "on prompts, or drop the training-data restriction)"
)
"""Named for the one guardrail reason the feedback called out: it reads as an account fact
with no way to act on it, and the account's own privacy settings are exactly that way."""


@dataclass(frozen=True, slots=True)
class CheckPlan:
    """The availability pre-check a sweep planned: the ranked candidates, the cut, the pins.

    The estimate is rendered before the check runs, so it prices this plan rather than what
    the check found: every ranked candidate is visited in ranking order — the check only
    stops early once `keep` of them have survived — and each visit is one smallest-rung
    request priced at that candidate's listed input price. `keep` is `--top`, and it is also
    what the estimate narrows its ranked spec rows to.

    `pinned` is checked too, but never against `keep`: a `--pin` is not competing with the
    ranked candidates for a slot, so its own failure frees none of theirs, and a ranked
    candidate's failure never promotes a pin early. Keeping the two lists apart is what makes
    that true — a single combined list, cut by `keep` in list order, would stop before ever
    reaching a pin placed after enough healthy ranked candidates.
    """

    candidates: Sequence[RunSpec]
    keep: int | None = None
    pinned: Sequence[RunSpec] = ()


@dataclass(frozen=True, slots=True)
class PreCheckResult:
    """One candidate as the pre-check left it: kept, or removed with the reason why."""

    spec: RunSpec
    kept: bool
    status: int = 0
    latency_ms: float = 0.0
    reason: str | None = None
    record: ProbeResult | None = None
    """The request the check sent, role `precheck`; written to `precheck.jsonl`, never
    folded into the spec's own records or its hit/error statistics."""


async def precheck_candidates(
    candidates: Sequence[RunSpec],
    entries: Sequence[TraceEntry],
    options: ProbeOptions,
    env: Mapping[str, str],
    *,
    keep: int | None = None,
    pinned: Sequence[RunSpec] = (),
    on_result: Callable[[PreCheckResult], None] | None = None,
) -> list[PreCheckResult]:
    """Check ranked candidates in order, stopping once `keep` have survived, then every pin.

    Sequential by construction: whether to send the next ranked request depends on the ones
    already sent, and a failure has to free the slot it was checked for rather than fill it.
    A pin is visited unconditionally after the ranked candidates, regardless of `keep`: it
    never competed for a ranked slot, so neither its own failure nor a ranked one changes
    whether it gets checked.
    """
    rungs = options.rungs or []
    if not rungs:
        raise InvalidInput("The availability pre-check needs a rung to send")
    rung = min(rungs)
    entry = entries[rung - 1]
    try:
        keys = {spec.label: resolve_api_key(spec.target, env) for spec in (*candidates, *pinned)}
    except ValueError as exc:
        raise InvalidInput(str(exc)) from exc
    timeout = httpx.Timeout(options.timeout_s, connect=_CONNECT_TIMEOUT_S)
    results: list[PreCheckResult] = []

    async def visit(spec: RunSpec) -> PreCheckResult:
        result = await _check(
            client, spec, entry=entry, options=options, api_key=keys[spec.label], rung=rung
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
        return result

    async with httpx.AsyncClient(timeout=timeout) as client:
        for spec in candidates:
            await visit(spec)
            if keep is not None and sum(1 for checked in results if checked.kept) >= keep:
                break
        for spec in pinned:
            await visit(spec)
    return results


async def _check(
    client: httpx.AsyncClient,
    spec: RunSpec,
    *,
    entry: TraceEntry,
    options: ProbeOptions,
    api_key: str,
    rung: int,
) -> PreCheckResult:
    """Send one candidate's smallest-rung body and read the answer."""
    responses: list[httpx.Response] = []

    async def capture(
        client_: httpx.AsyncClient,
        spec_: RunSpec,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        timeout_s: float,
    ) -> httpx.Response:
        """`post`, keeping the response so the reason can quote the body the record drops."""
        response = await post(client_, spec_, headers, body, timeout_s=timeout_s)
        responses.append(response)
        return response

    probe = ProbeContext(
        spec=spec, opts=options.replay_options(), api_key=api_key, client=client, throughput=False
    )
    body = prepare_body(entry.body, spec, probe.opts)
    outcome = await send(probe, capture, entry, body, ProbeCall(rung, "precheck", 0, None))
    record = outcome.record
    if is_served(record):
        return PreCheckResult(
            spec=spec, kept=True, status=record.status, latency_ms=record.latency_ms, record=record
        )
    return PreCheckResult(
        spec=spec,
        kept=False,
        status=record.status,
        latency_ms=record.latency_ms,
        reason=unavailable_reason(
            status=record.status, payload=_payload_of(responses), error=record.error
        ),
        record=record,
    )


def unavailable_reason(*, status: int, payload: object, error: str | None = None) -> str:
    """Why a candidate cannot be probed, in the API's own words where it gave any.

    The two documented cases read differently because they mean different things: a 404 from
    a pinned endpoint is the account's settings talking (`unavailable for this key`, naming
    the restriction), while an overload that survived the retries is the provider's.

    The kind is read off the record's own error first, with the same extraction the probe's
    `skipped, <error_type>` note uses: a 429 the gateway wraps in its own envelope names the
    failure in the wrapped object, and a reader comparing the two lines would otherwise be
    told the same refusal was `error` here and `rate_limit_exceeded` there. The response body
    is the fallback, for a record whose error is a transport message rather than a payload.
    """
    kind = error_type(error) or _body_error_type(payload)
    if status == _NOT_FOUND or kind == "not_found":
        detail = _first_reason(_message(payload) or error or "") or kind or "not_found"
        reason = f"unavailable for this key: {detail[:_REASON_CAP]}"
        if "training" in detail.lower():
            reason += _PRIVACY_HINT
        return reason
    if kind is not None:
        return f"unavailable: {kind}"
    if status:
        return f"unavailable: HTTP {status}"
    return f"unavailable: {(error or 'no response')[:_REASON_CAP]}"


def _first_reason(message: str) -> str | None:
    """The first reason the API lists, without the `: 1 endpoint excluded` tail.

    The message is "…We removed them for the following reasons (an endpoint may have matched
    multiple reasons):" and then one line per reason; the first of those lines is the one the
    account has to act on, and its own first colon introduces the count.
    """
    lines = message.splitlines()
    header = next((index for index, line in enumerate(lines) if "reasons" in line), None)
    if header is None:
        return None
    tail = lines[header][lines[header].index("reasons") + len("reasons") :].lstrip()
    if tail.startswith("(") and ")" in tail:
        tail = tail[tail.index(")") + 1 :]
    text = "\n".join([tail.lstrip(": ").strip(), *lines[header + 1 :]]).strip()
    line = text.splitlines()[0].strip() if text else ""
    if not line:
        return None
    return line.split(":", 1)[0].strip() or line


def _payload_of(responses: Sequence[httpx.Response]) -> object:
    """The last response body as JSON, or `None` when there is none or it is not JSON."""
    if not responses:
        return None
    try:
        return responses[-1].json()
    except ValueError:
        return None


def _message(payload: object) -> str | None:
    """The message an error body carries, at the top level or inside its `error` object."""
    for source in (payload, _field(payload, "error")):
        value = _field(source, "message")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _body_error_type(payload: object) -> str | None:
    """The response body's name for the failure, when the record's own error carries none.

    Read the same way the record's error is: the word `error` is a wrapper's marker rather
    than a kind, so a body that spells only that names nothing and the caller falls back to
    the status, which is at least true of the response.
    """
    for source in (payload, _field(payload, "error")):
        for key in ("error_type", "type", "code"):
            value = _field(source, key)
            if isinstance(value, str) and value and value != "error":
                return value
    return None


def _field(value: object, key: str) -> object:
    if not isinstance(value, dict):
        return None
    return cast("dict[str, Any]", value).get(key)
