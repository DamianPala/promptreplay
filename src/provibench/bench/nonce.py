"""Cache nonces: one run-wide hex, stamped into the first system block of every request.

A provider's prefix cache keys on the exact prompt bytes, so a replayed trace would
otherwise read back whatever an earlier run — or the operator's live session — left
behind. Stamping a fresh nonce gives every run its own cache namespace: a cold request is
cold because nobody has ever sent those bytes, not because we hoped the cache expired.

A probe run uses one nonce per rung and endpoint (`provibench-probe:<run hex>:<rung>:<endpoint>`),
so rung 30 does not read back what rung 1 wrote and one endpoint never reads back what
another endpoint wrote; a full replay uses one nonce for the whole run, so turn 2 reads what
turn 1 wrote, which is the effect being measured.
"""

from __future__ import annotations

import copy
import secrets
from collections.abc import Iterable
from typing import Any, cast

PROBE_PREFIX = "provibench-probe:"
RUN_PREFIX = "provibench-run:"


def new_run_hex() -> str:
    """The run's shared hex; every nonce of the run derives from it."""
    return secrets.token_hex(6)


def run_nonce(run_hex: str) -> str:
    """The single nonce a full replay stamps into every turn."""
    return f"{RUN_PREFIX}{run_hex}"


def probe_nonce(run_hex: str, rung: int, endpoint: str) -> str:
    """The nonce for one probe rung of one endpoint.

    `endpoint` is the spec label (`target:model@provider`) as `ProbeContext`/`probe.py` know
    it: plain ASCII with `:`/`@`/`/`, so it needs no encoding. Two OpenRouter tags of one
    provider probed in the same run are otherwise indistinguishable requests sent one after
    another, and the second one's "cold" write would be a full cache hit of the first.
    """
    return f"{PROBE_PREFIX}{run_hex}:{rung}:{endpoint}"


def has_system_text(body: dict[str, Any]) -> bool:
    """Whether `inject_nonce` can stamp this body at all; a cheap pre-flight check."""
    system = body.get("system")
    if isinstance(system, str):
        return True
    if not isinstance(system, list) or not system:
        return False
    first = cast("list[Any]", system)[0]
    return hasattr(first, "get") and isinstance(first.get("text"), str)


def require_stampable(labelled: Iterable[tuple[str, dict[str, Any]]]) -> None:
    """Raise for the first body the run nonce cannot be stamped into, if any.

    Called before a run sends anything: a request that cannot be isolated would silently
    measure whatever cache the provider already has, which is the measurement this exists
    to avoid. `--warm` is the way to ask for that measurement on purpose.
    """
    label = next((label for label, body in labelled if not has_system_text(body)), None)
    if label is None:
        return
    raise ValueError(
        f"{label} has no system prompt to stamp the run nonce into; "
        "--warm measures the cache as found instead"
    )


def inject_nonce(body: dict[str, Any], nonce: str) -> dict[str, Any]:
    """A copy of `body` with the nonce prepended to its first system block.

    `cache_control` and every other field are left untouched, so what the nonce changes is
    only the cached prefix. A body the nonce cannot be stamped into raises rather than
    silently running unisolated.
    """
    out = copy.deepcopy(body)
    system = out.get("system")
    if system is None:
        raise ValueError("body has no 'system' prompt to stamp the nonce into")
    if isinstance(system, str):
        out["system"] = f"{nonce}\n{system}"
        return out
    if not isinstance(system, list) or not system:
        kind = type(cast("object", system)).__name__
        raise ValueError(
            f"body['system'] must be a string or a non-empty list of blocks, got {kind}"
        )
    blocks = cast("list[Any]", system)
    # `hasattr` rather than `isinstance(dict)` keeps the element typed `Any`, so the
    # block's own `cache_control` and siblings are never re-typed or reordered.
    if not hasattr(blocks[0], "get") or not isinstance(blocks[0].get("text"), str):
        raise ValueError("body['system'][0] has no 'text' block to stamp the nonce into")
    blocks[0]["text"] = f"{nonce}\n{blocks[0]['text']}"
    return out
