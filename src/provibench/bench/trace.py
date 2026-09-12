"""Trace file format: one JSON line per recorded API request."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Anthropic usage semantics: input_tokens excludes the two cache buckets."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def prompt_total(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> Usage:
        raw = raw or {}
        return cls(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        )


class RecordedResponse(BaseModel):
    status: int
    latency_ms: float
    ttft_ms: float | None = None
    message_id: str | None = None
    model: str | None = None
    provider: str | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


class TraceEntry(BaseModel):
    seq: int
    ts: str
    path: str
    query: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any]
    conversation: str
    response: RecordedResponse | None = None

    @property
    def body_bytes(self) -> int:
        return len(json.dumps(self.body, ensure_ascii=False))


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for raw_block in cast("list[Any]", content):
            if isinstance(raw_block, dict):
                block = cast("dict[str, Any]", raw_block)
                if block.get("type") == "text":
                    return str(block.get("text", ""))
    return ""


def conversation_key(body: dict[str, Any]) -> str:
    """Identify a conversation by its first user message.

    A harness interleaves background calls (title generation, summarisation) with
    the main loop; they share the model but start from different first messages.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "empty"
    for raw_msg in cast("list[Any]", messages):
        if isinstance(raw_msg, dict):
            msg = cast("dict[str, Any]", raw_msg)
            if msg.get("role") == "user":
                seed = _text_of(msg.get("content"))[:4000]
                return hashlib.sha1(seed.encode("utf-8", "replace")).hexdigest()[:12]
    return "empty"


def append_entry(path: Path, entry: TraceEntry) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json(exclude_none=True) + "\n")


def load_trace(path: Path) -> list[TraceEntry]:
    entries: list[TraceEntry] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(TraceEntry.model_validate_json(line))
    return entries


def group_conversations(entries: list[TraceEntry]) -> dict[str, list[TraceEntry]]:
    groups: dict[str, list[TraceEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.conversation].append(entry)
    return dict(groups)


def main_conversation(entries: list[TraceEntry]) -> list[TraceEntry]:
    """The conversation carrying the most request bytes: the agent's main loop."""
    groups = group_conversations(entries)
    if not groups:
        return []
    return max(groups.values(), key=lambda g: sum(e.body_bytes for e in g))
