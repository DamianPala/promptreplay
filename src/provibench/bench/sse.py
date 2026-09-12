"""Extract message metadata and usage from Anthropic Messages API responses."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, Field

from provibench.bench.trace import Usage


class ParsedMessage(BaseModel):
    message_id: str | None = None
    model: str | None = None
    provider: str | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


def _merge_usage(base: dict[str, Any], extra: dict[str, Any] | None) -> None:
    for key, value in (extra or {}).items():
        if isinstance(value, int):
            base[key] = value


def parse_json_message(data: dict[str, Any]) -> ParsedMessage:
    if data.get("type") == "error" or "error" in data:
        err = data.get("error", data)
        return ParsedMessage(error=json.dumps(err)[:500])
    return ParsedMessage(
        message_id=data.get("id"),
        model=data.get("model"),
        provider=data.get("provider"),
        stop_reason=data.get("stop_reason"),
        usage=Usage.from_api(data.get("usage")),
    )


def _as_dict(value: object) -> dict[str, Any]:
    """`value` as a string-keyed dict, or an empty one when it isn't."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _apply_event(out: ParsedMessage, usage: dict[str, Any], data: dict[str, Any]) -> None:
    kind = data.get("type")
    if kind == "message_start":
        msg = _as_dict(data.get("message"))
        out.message_id = msg.get("id")
        out.model = msg.get("model")
        out.provider = msg.get("provider") or data.get("provider")
        _merge_usage(usage, msg.get("usage"))
    elif kind == "message_delta":
        delta = _as_dict(data.get("delta"))
        out.stop_reason = delta.get("stop_reason") or out.stop_reason
        _merge_usage(usage, data.get("usage"))
    elif kind == "error":
        out.error = json.dumps(data.get("error", data))[:500]


def parse_event_stream(raw: bytes) -> ParsedMessage:
    """Fold an SSE stream into one ParsedMessage.

    message_start carries id/model and the prompt-side usage; message_delta carries
    output_tokens (and on some gateways a final input_tokens correction).
    """
    out = ParsedMessage()
    usage: dict[str, Any] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            raw_data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        data = _as_dict(raw_data)
        if data:
            _apply_event(out, usage, data)
    out.usage = Usage.from_api(usage)
    return out


def parse_response(content_type: str, raw: bytes) -> ParsedMessage:
    if "text/event-stream" in content_type:
        return parse_event_stream(raw)
    try:
        return parse_json_message(json.loads(raw or b"{}"))
    except json.JSONDecodeError:
        return ParsedMessage(error=raw[:500].decode("utf-8", "replace"))
