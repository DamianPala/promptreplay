"""The F3 error envelope as one exception hierarchy.

Every failure the tool reports is an instance of `CliError`. The entry point maps the
exception to its exit code and to the F3 object on the last stderr line; nothing else
calls `sys.exit` or prints errors.
"""

from enum import StrEnum
from typing import ClassVar

from provibench.core.documents import Document


class ErrorKind(StrEnum):
    """F3 `kind` values with shared meanings that this tool uses."""

    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNAUTHENTICATED = "unauthenticated"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    OUTCOME_UNKNOWN = "outcome_unknown"
    INTERRUPTED = "interrupted"
    CURSOR_UNAVAILABLE = "cursor_unavailable"
    CONFIRMATION_REQUIRED = "confirmation_required"
    OPERATION_FAILED = "operation_failed"
    PRECONDITION_FAILED = "precondition_failed"


class Action(StrEnum):
    """F3 `action`: who can recover from the failure."""

    AGENT = "agent"
    USER = "user"
    NONE = "none"


class CliError(Exception):
    """A failure with a stable `kind`, an exit code, and the optional F3 fields.

    Subclasses fix `kind` and `exit_code`; a tool-defined kind subclasses this directly.
    """

    kind: ClassVar[str]
    exit_code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        context: Document | None = None,
        retryable: bool | None = None,
        action: Action | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.context = context
        self.retryable = retryable
        self.action = action

    def to_document(self) -> Document:
        """The F3 envelope: exactly one top-level field, `error`."""
        error: Document = {"kind": self.kind, "message": self.message}
        if self.retryable is not None:
            error["retryable"] = self.retryable
        if self.action is not None:
            error["action"] = self.action.value
        if self.hint is not None:
            error["hint"] = self.hint
        if self.context is not None:
            error["context"] = self.context
        return {"error": error}


class InvalidInput(CliError):
    """A usage error or an input outside an I7 bound; exits 2."""

    kind = ErrorKind.INVALID_INPUT
    exit_code = 2


class ConfirmationRequired(CliError):
    """R3 stopped the call before its gated effect; exits 2."""

    kind = ErrorKind.CONFIRMATION_REQUIRED
    exit_code = 2


class NotFound(CliError):
    """A target named by an argument or flag does not exist."""

    kind = ErrorKind.NOT_FOUND


class Conflict(CliError):
    """Authoritative state conflicts with the requested change."""

    kind = ErrorKind.CONFLICT


class Unauthenticated(CliError):
    """The caller could not be identified."""

    kind = ErrorKind.UNAUTHENTICATED


class Timeout(CliError):
    """The command deadline passed; any effect is absent or identified in context."""

    kind = ErrorKind.TIMEOUT


class Unavailable(CliError):
    """A dependency failed the request transiently; retryable unless said otherwise."""

    kind = ErrorKind.UNAVAILABLE

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        hint: str | None = None,
        context: Document | None = None,
    ) -> None:
        super().__init__(message, hint=hint, context=context, retryable=retryable)


class OutcomeUnknown(CliError):
    """The intended effect may have happened and was not observed."""

    kind = ErrorKind.OUTCOME_UNKNOWN


class Interrupted(CliError):
    """The user interrupted the command."""

    kind = ErrorKind.INTERRUPTED


class CursorUnavailable(CliError):
    """A cursor is malformed, incompatible, or points at history that is gone."""

    kind = ErrorKind.CURSOR_UNAVAILABLE


class OperationFailed(CliError):
    """A managed operation reached a terminal state other than `succeeded`."""

    kind = ErrorKind.OPERATION_FAILED


class PreconditionFailed(CliError):
    """A documented precondition that `--force` overrides is not met."""

    kind = ErrorKind.PRECONDITION_FAILED
