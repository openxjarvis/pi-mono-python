from __future__ import annotations

from typing import Any


class PiServerError(Exception):
    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message", "Pi server error"))
        self.name = "PiServerError"
        self.code = error.get("code")
        self.details = error.get("details")


class PiDisconnectedError(Exception):
    def __init__(self, message: str = "Pi client is disconnected") -> None:
        super().__init__(message)
        self.name = "PiDisconnectedError"


class PiClientDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Pi client is disposed")
        self.name = "PiClientDisposedError"


class PiSessionOwnershipError(Exception):
    def __init__(self, session_id: str, message: str) -> None:
        super().__init__(message)
        self.name = "PiSessionOwnershipError"
        self.session_id = session_id


class PiSessionDetachedError(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id} is not attached")
        self.name = "PiSessionDetachedError"
        self.session_id = session_id


def to_error(error: object) -> Exception:
    return error if isinstance(error, Exception) else Exception(str(error))


def to_disconnected_error(error: object) -> PiDisconnectedError:
    cause = to_error(error)
    return cause if isinstance(cause, PiDisconnectedError) else PiDisconnectedError(str(cause))
