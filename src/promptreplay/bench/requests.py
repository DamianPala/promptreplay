"""The HTTP plumbing `replay` and `probe` share: headers, the POST, parsing, the retries.

Both protocols must send byte-identical requests down the same code path, or their numbers
are not comparable; keeping the helpers here is what makes that literal.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from promptreplay.bench.sse import ParsedMessage, parse_json_message
from promptreplay.bench.targets import RunSpec
from promptreplay.bench.trace import TraceEntry

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_THINKING_RETRY_TRIGGERS = ("max_tokens", "budget", "thinking")


def should_retry_without_thinking(body_text: str) -> bool:
    """Whether a 400 is the `thinking` parameter meeting a one-token `max_tokens`.

    Replay and probe both send `max_tokens: 1`, so a recorded `thinking` parameter makes the
    request impossible; dropping it and resending the same bytes minus that one key is the
    retry. The match is deliberately loose because the wording differs per gateway.
    """
    lowered = body_text.lower()
    return any(trigger in lowered for trigger in _THINKING_RETRY_TRIGGERS)


def build_headers(entry: TraceEntry, api_key: str) -> dict[str, str]:
    """The recorded request's headers, re-authenticated for this run."""
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "anthropic-version": entry.headers.get("anthropic-version", DEFAULT_ANTHROPIC_VERSION),
    }
    beta = entry.headers.get("anthropic-beta")
    if beta:
        headers["anthropic-beta"] = beta
    return headers


def parse_body(resp: httpx.Response) -> ParsedMessage:
    """The response's message metadata and usage; never raises on a non-JSON body."""
    try:
        data = resp.json()
    except ValueError:
        return ParsedMessage(error=resp.text[:500])
    if not isinstance(data, dict):
        return ParsedMessage(error=str(data)[:500])
    return parse_json_message(cast("dict[str, Any]", data), status=resp.status_code)


async def post(
    client: httpx.AsyncClient,
    spec: RunSpec,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout_s: float,
) -> httpx.Response:
    """POST one prepared body to the spec's target."""
    return await client.post(spec.target.url, headers=headers, json=body, timeout=timeout_s)
