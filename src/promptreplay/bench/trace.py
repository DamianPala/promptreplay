"""Trace file format: one JSON line per recorded API request.

Recording always writes plain jsonl; reading accepts a `.gz` path transparently, so a
shared (scrubbed, compressed) trace works everywhere a recorded one does.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

SAMPLE_TRACE = "sample-trace"
"""The literal TRACE argument that resolves to the packaged example trace."""

PACKAGED_SAMPLE = "sample-trace.jsonl.gz"
"""The packaged example trace, next to `targets.toml` under `promptreplay.data`."""

TRACE_SUFFIXES = ("jsonl", "gz")
"""Suffixes a trace file name may carry, stripped to give the trace's name."""

NAME_FORMS = ("", ".jsonl", ".jsonl.gz", ".gz")
"""What a TRACE name may be missing under `traces_dir`, in the order they are tried."""


class TraceError(Exception):
    """A trace cannot be resolved, read, or parsed; commands map this to an F3 error."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class TraceNotFound(TraceError):
    """The TRACE argument names no file."""


class PackagedSampleMissing(TraceError):
    """The packaged `sample-trace` is not in the installed distribution."""


class TraceParseError(TraceError, ValueError):
    """One line of a trace is not a valid entry."""

    def __init__(self, line: int, detail: str) -> None:
        super().__init__(f"Trace line {line} is not a valid entry: {detail}")
        self.line = line
        self.detail = detail


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


def trace_name(path: Path) -> str:
    """A trace's name: its file name without the `.jsonl` and `.gz` suffixes.

    `traces/sample-trace.jsonl.gz`, `traces/sample-trace.jsonl` and `traces/sample-trace.gz`
    are all the trace `sample-trace`, which is the name runs are filed under.
    """
    name = path.name
    for suffix in reversed(TRACE_SUFFIXES):
        ending = f".{suffix}"
        if name.endswith(ending):
            name = name[: -len(ending)]
    return name or path.stem


def dump_document(document: dict[str, Any]) -> str:
    """One entry as the compact JSON line `append_entry` writes."""
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def append_entry(path: Path, entry: TraceEntry) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(dump_document(entry.model_dump(exclude_none=True, mode="json")) + "\n")


def read_trace_text(path: Path) -> str:
    """A whole trace as UTF-8 text; a `.gz` path is decompressed transparently."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


def write_trace_text(path: Path, text: str) -> None:
    """Write a trace; a `.gz` path is compressed with a fixed header, so bytes repeat.

    `gzip.compress(mtime=0)` leaves the timestamp out of the header: without it the same
    input and options would produce different bytes on every run.
    """
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        return
    path.write_text(text, encoding="utf-8")


def parse_trace_text(text: str) -> list[dict[str, Any]]:
    """The entry documents in a trace's text; a malformed line is a `TraceParseError`."""
    return [document for _, document in _numbered_documents(text)]


def load_trace(path: Path) -> list[TraceEntry]:
    entries: list[TraceEntry] = []
    for number, document in _numbered_documents(read_trace_text(path)):
        try:
            entries.append(TraceEntry.model_validate(document))
        except ValidationError as exc:
            raise TraceParseError(number, _validation_detail(exc)) from exc
    return entries


def sample_trace_path() -> Path:
    """The packaged example trace, next to the packaged `targets.toml`."""
    resource = resources.files("promptreplay.data").joinpath(PACKAGED_SAMPLE)
    path = Path(str(resource))
    if not path.is_file():
        raise PackagedSampleMissing(
            f"The packaged trace 'sample-trace' is missing at {path}",
            hint=(
                "Reinstall promptreplay; the distribution ships "
                "promptreplay/data/sample-trace.jsonl.gz"
            ),
        )
    return path


def resolve_trace(arg: str, traces_dir: Path, cwd: Path | None = None) -> Path:
    """TRACE as an existing path, a name under `traces_dir`, or the packaged `sample-trace`.

    A trace of the caller's own named `sample-trace` wins over the packaged example, so a
    working directory that happens to hold one keeps inspecting its own file.
    """
    candidate = Path(arg)
    if not candidate.is_absolute():
        candidate = (cwd or Path.cwd()) / candidate
    if candidate.is_file():
        return candidate
    for form in NAME_FORMS:
        named = traces_dir / f"{arg}{form}"
        if named.is_file():
            return named
    if arg == SAMPLE_TRACE:
        return sample_trace_path()
    raise TraceNotFound(
        f"No trace file at {candidate} or {traces_dir / f'{arg}.jsonl'}",
        hint="Pass an existing path, a name under traces_dir, or 'sample-trace'",
    )


def _numbered_documents(text: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Each non-empty line of a trace, with its 1-based line number."""
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceParseError(number, f"invalid JSON ({exc.msg})") from exc
        if not isinstance(value, dict):
            raise TraceParseError(number, "not a JSON object")
        yield number, cast("dict[str, Any]", value)


def _validation_detail(error: ValidationError) -> str:
    """The first pydantic failure as `field: message`, for the parse error's line."""
    failure = error.errors()[0]
    location = ".".join(str(part) for part in failure["loc"]) or "(entry)"
    return f"{location}: {failure['msg']}"


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


def total_prompt_tokens(entries: Sequence[TraceEntry]) -> int:
    """The sum of every turn's own recorded prompt size: the bill of replaying them all once.

    Each turn already resends the whole conversation so far, so its `prompt_total` is what
    that one turn costs; summing them is what a session like this trace bills in prompt
    tokens end to end, cache hits aside. A turn with no recorded response contributes
    nothing rather than making the total unknown -- a partially recorded trace still gives
    a number, understated by exactly what it is missing.
    """
    return sum(entry.response.usage.prompt_total for entry in entries if entry.response)
