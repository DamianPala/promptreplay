"""`inspect`: list a trace's conversations and show one conversation's turns.

`provibench.bench.trace` pulls in pydantic; importing it only inside the callback keeps
building the CLI (schema, --help, completion) cheap. `resolve_trace_path` is also used by
`provibench.commands.replay`.
"""

from __future__ import annotations

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
from provibench.core.errors import NotFound
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
    },
    required=["key", "requests", "bytes", "first_seq", "last_seq", "prompt_first", "prompt_last"],
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

_CONVERSATION_COLUMNS = ("key", "requests", "bytes", "first_seq", "last_seq", "prompt_first")
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


def render_inspect_table(invocation: Invocation, document: Document) -> None:
    """Two tables: every conversation, then the selected one's turns."""
    from rich import box
    from rich.markup import escape
    from rich.table import Table

    def cell(value: object) -> str:
        return escape(escape_terminal_text("-" if value is None else str(value)))

    console = invocation.stdout_console()
    selected = document.get("selected")
    conversations = [d for d in map(as_document, as_list(document.get("conversations")) or []) if d]
    conv_table = Table(box=box.SIMPLE, header_style="bold", title="Conversations")
    for column in _CONVERSATION_COLUMNS:
        conv_table.add_column(column)
    for entry in conversations:
        marker = " *" if entry.get("key") == selected else ""
        cells = [cell(entry.get(column)) for column in _CONVERSATION_COLUMNS]
        cells[0] += marker
        conv_table.add_row(*cells)
    console.print(conv_table)

    turns = [d for d in map(as_document, as_list(document.get("turns")) or []) if d]
    turn_table = Table(box=box.SIMPLE, header_style="bold", title=f"Turns ({selected or '-'})")
    for column in _TURN_COLUMNS:
        turn_table.add_column(column)
    for entry in turns:
        turn_table.add_row(*(cell(entry.get(column)) for column in _TURN_COLUMNS))
    console.print(turn_table)


@click.command(
    "inspect",
    cls=Command,
    spec=CommandSpec(effects=Effects.READ_ONLY, output=_OUTPUT, render=render_inspect_table),
    help="List a trace's conversations and show one conversation's turns.\n\n"
    "TRACE is either an existing path or a name under traces_dir. Without "
    "--conversation, the conversation carrying the most request bytes (the agent's "
    "main loop) is selected.",
)
@click.argument("trace")
@click.option(
    "--conversation", default=None, help="Conversation key to inspect; defaults to the main one"
)
@click.pass_context
def inspect(ctx: click.Context, trace: str, conversation: str | None) -> Document:
    from provibench.bench.trace import group_conversations, load_trace

    invocation = require_invocation(ctx)
    trace_path = resolve_trace_path(trace, invocation)
    entries = load_trace(trace_path)
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
    return {
        "key": key,
        "requests": len(ordered),
        "bytes": sum(e.body_bytes for e in ordered),
        "first_seq": ordered[0].seq,
        "last_seq": ordered[-1].seq,
        "prompt_first": _preview(ordered[0]),
        "prompt_last": _preview(ordered[-1]),
    }


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
    """TRACE as an existing path, or as `<traces_dir>/<trace>.jsonl`."""
    candidate = Path(trace)
    if not candidate.is_absolute():
        candidate = invocation.cwd / candidate
    if candidate.is_file():
        return candidate
    traces_dir = Path(invocation.setting("traces_dir") or ".")
    named = traces_dir / f"{trace}.jsonl"
    if named.is_file():
        return named
    raise NotFound(
        f"No trace file at {candidate} or {named}",
        hint="Pass an existing path, or a name under traces_dir",
    )
