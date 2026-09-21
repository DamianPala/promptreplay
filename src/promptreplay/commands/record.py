"""`record`: run the recording proxy in front of an upstream and summarise what it captured.

`promptreplay.bench.proxy` pulls in uvicorn, starlette, and httpx; importing it only inside
the callback keeps building the CLI (schema, --help, completion) cheap, matching how
`core.version` defers `importlib.metadata` and `core.context` defers `rich`.
"""

import contextlib
from pathlib import Path

import click

from promptreplay.core.documents import Document, array, boolean, integer, obj, string
from promptreplay.core.errors import InvalidInput
from promptreplay.core.params import TIMEOUT
from promptreplay.core.registry import Command, require_invocation
from promptreplay.core.spec import CommandSpec, Effects

_CONVERSATION = obj(
    {"key": string(), "requests": integer(), "bytes": integer()},
    required=["key", "requests", "bytes"],
)
_OUTPUT = obj(
    {
        "trace": string(),
        "requests": integer(),
        "conversations": array(_CONVERSATION),
        "changed": boolean(),
    },
    required=["trace", "requests", "conversations", "changed"],
)


@click.command(
    "record",
    cls=Command,
    spec=CommandSpec(effects=Effects.NON_IDEMPOTENT, output=_OUTPUT),
    help="Record API traffic through a reverse proxy until interrupted.\n\n"
    "Starts a recording proxy in front of --upstream and appends every "
    "POST .../v1/messages exchange to <traces-dir>/<name>.jsonl. Point the harness at "
    "the proxy (for example ANTHROPIC_BASE_URL=http://127.0.0.1:8787) and press Ctrl-C "
    "to stop; the trace is then summarised by conversation. While it waits it logs one line "
    "per recorded request and no periodic heartbeat, even under --verbose.",
)
@click.option("--name", required=True, help="Trace name; written to <traces-dir>/<name>.jsonl")
@click.option("--upstream", required=True, help="Upstream base URL to forward requests to")
@click.option("--host", default="127.0.0.1", help="Address to listen on")
@click.option("--port", type=click.IntRange(1, 65535), default=8787, help="Port to listen on")
@click.option("--append", is_flag=True, help="Append to an existing trace file instead of refusing")
@click.option(
    "--timeout",
    type=TIMEOUT,
    default=None,
    metavar="DURATION",
    help="Stop after this duration; the wait is unbounded by default",
)
@click.pass_context
def record(
    ctx: click.Context,
    *,
    name: str,
    upstream: str,
    host: str,
    port: int,
    append: bool,
    timeout: float | None,
) -> Document:
    from promptreplay.bench.proxy import serve
    from promptreplay.bench.trace import group_conversations, load_trace

    invocation = require_invocation(ctx)
    traces_dir = Path(invocation.setting("traces_dir") or ".")
    trace_path = traces_dir / f"{name}.jsonl"
    if trace_path.exists() and not append:
        raise InvalidInput(
            f"Trace file {trace_path} already exists",
            hint="Pass --append to add to it, or choose a different --name",
        )
    traces_dir.mkdir(parents=True, exist_ok=True)
    before = len(load_trace(trace_path)) if trace_path.exists() else 0

    invocation.message(
        f"recording {upstream} -> {trace_path}  listen http://{host}:{port}  "
        "(point the harness at it; Ctrl-C to stop)"
    )
    with contextlib.suppress(KeyboardInterrupt):
        serve(upstream, trace_path, host, port, log=invocation.message, timeout_s=timeout)

    entries = load_trace(trace_path) if trace_path.exists() else []
    groups = group_conversations(entries)
    conversations = [
        {"key": key, "requests": len(group), "bytes": sum(e.body_bytes for e in group)}
        for key, group in groups.items()
    ]
    return {
        "trace": str(trace_path),
        "requests": len(entries),
        "conversations": conversations,
        "changed": len(entries) > before,
    }
