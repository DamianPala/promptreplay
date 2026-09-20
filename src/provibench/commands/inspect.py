"""`inspect`: list a trace's conversations and show one conversation's turns.

`provibench.bench.trace` pulls in pydantic; importing it only inside the callback keeps
building the CLI (schema, --help, completion) cheap. The TRACE helpers here —
`resolve_trace_path`, `load_trace_entries`, `load_trace_text`, `load_trace_documents` —
are shared with `provibench.commands.replay` and `provibench.commands.scrub`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from provibench.core.context import Invocation
from provibench.core.documents import (
    Document,
    array,
    as_document,
    as_list,
    integer,
    nullable_integer,
    nullable_number,
    nullable_string,
    obj,
    string,
)
from provibench.core.errors import Action, InvalidInput, NotFound, OperationFailed
from provibench.core.registry import Command, require_invocation
from provibench.core.spec import CommandSpec, Effects
from provibench.core.terminal_text import escape_terminal_text

if TYPE_CHECKING:
    from provibench.bench.trace import TraceEntry

_PREVIEW_LEN = 80

_CONVERSATION = obj(
    {
        "key": string(),
        "requests": integer(),
        "bytes": integer(),
        "first_seq": integer(),
        "last_seq": integer(),
        "prompt_first": nullable_string(),
        "prompt_last": nullable_string(),
        "prompt_tokens_first": nullable_integer(),
        "prompt_tokens_last": nullable_integer(),
    },
    required=[
        "key",
        "requests",
        "bytes",
        "first_seq",
        "last_seq",
        "prompt_first",
        "prompt_last",
        "prompt_tokens_first",
        "prompt_tokens_last",
    ],
)
_TURN = obj(
    {
        "seq": integer(),
        "turn": integer(),
        "status": nullable_integer(),
        "provider": nullable_string(),
        "prompt_total": nullable_integer(),
        "input": nullable_integer(),
        "cached": nullable_integer(),
        "cache_write": nullable_integer(),
        "output": nullable_integer(),
        "ttft_ms": nullable_number(),
        "latency_ms": nullable_number(),
    },
    required=[
        "seq",
        "turn",
        "status",
        "provider",
        "prompt_total",
        "input",
        "cached",
        "cache_write",
        "output",
        "ttft_ms",
        "latency_ms",
    ],
)
_OUTPUT = obj(
    {
        "trace": string(),
        "conversations": array(_CONVERSATION),
        "selected": nullable_string(),
        "turns": array(_TURN),
    },
    required=["trace", "conversations", "selected", "turns"],
)

_CONVERSATION_COLUMNS = ("key", "requests", "bytes", "prompt tokens")
_TURN_COLUMNS = (
    "turn",
    "seq",
    "status",
    "provider",
    "prompt_total",
    "cached",
    "cache_write",
    "output",
    "ttft_ms",
    "latency_ms",
)
_ARROW = " → "


def render_inspect_table(invocation: Invocation, document: Document) -> None:
    """Two sections, plain text like the rest of the tool: conversations, then selected turns.

    The text view trades the JSON's per-request detail for what fits a fixed-width table: a
    conversation's `first_seq`/`last_seq` and prompt-text previews stay JSON-only, replaced
    here by the prompt token count of its first and last request; a turns table with no
    provider on any row (a native trace) drops that column instead of printing it empty.
    """
    from provibench.bench.labels import text_table

    selected = document.get("selected")
    conversations = [d for d in map(as_document, as_list(document.get("conversations")) or []) if d]
    turns = [d for d in map(as_document, as_list(document.get("turns")) or []) if d]
    turn_columns = _TURN_COLUMNS
    if not any(entry.get("provider") is not None for entry in turns):
        turn_columns = tuple(column for column in _TURN_COLUMNS if column != "provider")
    lines = [
        "conversations:",
        *text_table(
            _CONVERSATION_COLUMNS, [_conversation_cells(entry, selected) for entry in conversations]
        ),
        "",
        f"turns: {selected or '-'}",
        *text_table(turn_columns, [_turn_cells(entry, turn_columns) for entry in turns]),
    ]
    stdout = invocation.streams.stdout
    stdout.write("\n".join(escape_terminal_text(line) for line in lines) + "\n")
    stdout.flush()


def _cell(value: object) -> str:
    return "-" if value is None else str(value)


def _conversation_cells(entry: Document, selected: object) -> list[str]:
    marker = " *" if entry.get("key") == selected else ""
    first = _cell(entry.get("prompt_tokens_first"))
    last = _cell(entry.get("prompt_tokens_last"))
    tokens = f"{first}{_ARROW}{last}"
    return [
        f"{_cell(entry.get('key'))}{marker}",
        _cell(entry.get("requests")),
        _cell(entry.get("bytes")),
        tokens,
    ]


def _turn_cells(entry: Document, columns: Sequence[str]) -> list[str]:
    return [_cell(entry.get(column)) for column in columns]


@click.command(
    "inspect",
    cls=Command,
    spec=CommandSpec(effects=Effects.READ_ONLY, output=_OUTPUT, render=render_inspect_table),
    help="List a trace's conversations and show one conversation's turns.\n\n"
    "TRACE is either an existing path, a name under traces_dir, or 'sample' (the "
    "packaged example). Without --conversation, the conversation carrying the most "
    "request bytes (the agent's main loop) is selected.",
)
@click.argument("trace", help="Trace path, a name under traces_dir, or 'sample'")
@click.option(
    "--conversation", default=None, help="Conversation key to inspect; defaults to the main one"
)
@click.pass_context
def inspect(ctx: click.Context, trace: str, conversation: str | None) -> Document:
    from provibench.bench.trace import group_conversations

    invocation = require_invocation(ctx)
    trace_path = resolve_trace_path(trace, invocation)
    entries = load_trace_entries(trace_path)
    groups = group_conversations(entries)
    conversations = sorted(
        (_conversation_document(key, group) for key, group in groups.items()),
        key=lambda c: str(c["key"]),
    )

    selected_key = _select_key(entries, groups, conversation)
    ordered = sorted(groups.get(selected_key, []), key=lambda e: e.seq) if selected_key else []
    turns = [_turn_document(turn, entry) for turn, entry in enumerate(ordered, start=1)]

    return {
        "trace": str(trace_path),
        "conversations": conversations,
        "selected": selected_key,
        "turns": turns,
    }


def _select_key(
    entries: list[TraceEntry], groups: dict[str, list[TraceEntry]], conversation: str | None
) -> str | None:
    from provibench.bench.trace import main_conversation

    if conversation is not None:
        if conversation not in groups:
            raise NotFound(f"No conversation {conversation!r} in this trace")
        return conversation
    main = main_conversation(entries)
    return main[0].conversation if main else None


def _conversation_document(key: str, group: list[TraceEntry]) -> Document:
    ordered = sorted(group, key=lambda e: e.seq)
    first, last = ordered[0], ordered[-1]
    return {
        "key": key,
        "requests": len(ordered),
        "bytes": sum(e.body_bytes for e in ordered),
        "first_seq": first.seq,
        "last_seq": last.seq,
        "prompt_first": _preview(first),
        "prompt_last": _preview(last),
        "prompt_tokens_first": _prompt_tokens(first),
        "prompt_tokens_last": _prompt_tokens(last),
    }


def _prompt_tokens(entry: TraceEntry) -> int | None:
    usage = entry.response.usage if entry.response else None
    return usage.prompt_total if usage else None


def _preview(entry: TraceEntry) -> str | None:
    messages = entry.body.get("messages")
    if not isinstance(messages, list):
        return None
    for raw_message in cast("list[Any]", messages):
        if isinstance(raw_message, dict):
            message = cast("dict[str, Any]", raw_message)
            if message.get("role") == "user":
                text = _text_of(message.get("content"))
                if text:
                    return text[:_PREVIEW_LEN]
    return None


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


def _turn_document(turn: int, entry: TraceEntry) -> Document:
    response = entry.response
    usage = response.usage if response else None
    return {
        "seq": entry.seq,
        "turn": turn,
        "status": response.status if response else None,
        "provider": response.provider if response else None,
        "prompt_total": usage.prompt_total if usage else None,
        "input": usage.input_tokens if usage else None,
        "cached": usage.cache_read_input_tokens if usage else None,
        "cache_write": usage.cache_creation_input_tokens if usage else None,
        "output": usage.output_tokens if usage else None,
        "ttft_ms": response.ttft_ms if response else None,
        "latency_ms": response.latency_ms if response else None,
    }


def resolve_trace_path(trace: str, invocation: Invocation) -> Path:
    """TRACE as an existing path, as `<traces_dir>/<trace>.jsonl[.gz]`, or the packaged sample."""
    from provibench.bench.trace import PackagedSampleMissing, TraceNotFound, resolve_trace

    traces_dir = Path(invocation.setting("traces_dir") or ".")
    try:
        return resolve_trace(trace, traces_dir, invocation.cwd)
    except TraceNotFound as exc:
        raise NotFound(str(exc), hint=exc.hint) from exc
    except PackagedSampleMissing as exc:
        raise OperationFailed(str(exc), hint=exc.hint, action=Action.NONE) from exc


def load_trace_entries(trace_path: Path) -> list[TraceEntry]:
    """A trace parsed into entries; unreadable or malformed input is an input error."""
    from provibench.bench.trace import load_trace

    return _read(lambda: load_trace(trace_path), trace_path)


def load_trace_text(trace_path: Path) -> str:
    """A trace's whole text; an unreadable file is an input error."""
    from provibench.bench.trace import read_trace_text

    return _read(lambda: read_trace_text(trace_path), trace_path)


def load_trace_documents(trace_path: Path) -> list[dict[str, Any]]:
    """A trace's entry documents, exactly as they were written.

    A line that is not JSON, or not an object, is an input error naming it. Entries are
    deliberately *not* validated against `TraceEntry`: `scrub` must never refuse to remove
    secrets from a trace, and writing the documents back verbatim is what keeps the copy
    byte-identical apart from the masked strings.
    """
    from provibench.bench.trace import parse_trace_text, read_trace_text

    return _read(lambda: parse_trace_text(read_trace_text(trace_path)), trace_path)


def _read[T](reader: Callable[[], T], trace_path: Path) -> T:
    """Run `reader`, reporting an unreadable trace or a bad line as an input error."""
    from provibench.bench.trace import TraceError

    try:
        return reader()
    except TraceError as exc:
        raise InvalidInput(str(exc)) from exc
    except OSError as exc:
        raise InvalidInput(f"Trace file {trace_path} could not be read: {exc}") from exc
