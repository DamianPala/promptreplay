"""What a command declares about itself beyond its parameters (R1, R3, I5b, O2, O4, O7)."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from provibench.core.context import Invocation
from provibench.core.documents import Document, JsonSchema
from provibench.core.output import STREAM_FORMAT_DEFAULTS, TOOL_FORMAT_DEFAULTS, FormatDefaults


class Effects(StrEnum):
    """R1 effect classes."""

    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"


type Renderer = Callable[[Invocation, Document], None]
"""Writes one success document in human-readable form; replaces the generic renderer."""

type Result = Document | Iterator[Document] | None
"""What a command callback returns: a document, a record stream, or nothing."""


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """The registry entry the parser dispatches with the command."""

    effects: Effects
    output: JsonSchema | None = None
    output_description: str | None = None
    confirm: bool = False
    interactive: bool = False
    stream: bool = False
    format_defaults: FormatDefaults | None = None
    plain_field: str | None = None
    render: Renderer | None = None
    listed: bool = True

    @property
    def formats(self) -> FormatDefaults:
        """The effective O2 defaults: the command's own, or the tool-wide ones."""
        if self.format_defaults is not None:
            return self.format_defaults
        return STREAM_FORMAT_DEFAULTS if self.stream else TOOL_FORMAT_DEFAULTS
