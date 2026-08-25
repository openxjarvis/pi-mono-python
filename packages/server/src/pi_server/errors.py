from __future__ import annotations

from typing import Any, Literal

PiServerOperationErrorCode = Literal["busy", "session_locked", "not_found", "invalid_request", "not_implemented"]

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"
NOT_IMPLEMENTED_MESSAGE = "Operation is not implemented"


class PiServerError(Exception):
    def __init__(self, code: PiServerOperationErrorCode, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.name = "PiServerError"
        self.code = code
        self.details = details


class SessionBusyError(PiServerError):
    def __init__(self, message: str = "Session is busy", details: Any = None) -> None:
        super().__init__("busy", message, details)
        self.name = "SessionBusyError"


class SessionLockedError(PiServerError):
    def __init__(self, message: str = "Session is locked", details: Any = None) -> None:
        super().__init__("session_locked", message, details)
        self.name = "SessionLockedError"


class SessionNotFoundError(PiServerError):
    def __init__(self, message: str = "Session was not found", details: Any = None) -> None:
        super().__init__("not_found", message, details)
        self.name = "SessionNotFoundError"


class NotImplementedError_(PiServerError):
    def __init__(self) -> None:
        super().__init__("not_implemented", NOT_IMPLEMENTED_MESSAGE)
        self.name = "NotImplementedError"


class InternalServerError(Exception):
    def __init__(self, cause: object) -> None:
        super().__init__(INTERNAL_SERVER_ERROR_MESSAGE)
        self.name = "InternalServerError"
        self.cause = cause


# Keep the TS export name without shadowing the builtin.
NotImplementedError = NotImplementedError_
