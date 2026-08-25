from __future__ import annotations

from typing import Any, Callable, TypeVar

from .cbor.decoder import decode_cbor
from .cbor.encoder import encode_cbor
from .framing import DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, assert_complete_frame, encode_frame
from .schemas import (
    PROTOCOL_VERSION,
    ClientMessage,
    ServerMessage,
    is_client_message,
    is_server_message,
)

T = TypeVar("T")


class ProtocolValidationError(Exception):
    def __init__(self, message: str, _value: object = None) -> None:
        super().__init__(message)
        self.name = "ProtocolValidationError"


def _is_protocol_value(value: object, optional_property: bool = False, ancestors: set[int] | None = None) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if not isinstance(value, (list, dict)):
        return optional_property
    ancestors = ancestors or set()
    ident = id(value)
    if ident in ancestors:
        return False
    ancestors.add(ident)
    try:
        if isinstance(value, list):
            return all(_is_protocol_value(item, False, ancestors) for item in value)
        return all(_is_protocol_value(item, True, ancestors) for item in value.values())
    finally:
        ancestors.discard(ident)


def parse_client_message(value: object) -> ClientMessage:
    if not _is_protocol_value(value) or not is_client_message(value):
        raise ProtocolValidationError("Invalid client protocol message")
    return value  # type: ignore[return-value]


def parse_server_message(value: object) -> ServerMessage:
    if not _is_protocol_value(value) or not is_server_message(value):
        raise ProtocolValidationError("Invalid server protocol message")
    return value  # type: ignore[return-value]


def _bounded_error_message(error: object) -> str:
    if not isinstance(error, Exception):
        return "Unknown codec error"
    message = str(error)
    return message if len(message) <= 500 else f"{message[:497]}..."


def _encode_protocol_message(
    value: T,
    parse: Callable[[object], T],
    kind: str,
    options: dict[str, int] | None = None,
) -> bytes:
    validated = parse(value)
    try:
        max_frame_length = (options or {}).get("maxFrameLength", (options or {}).get("max_frame_length", DEFAULT_MAX_FRAME_LENGTH))
        frame = encode_frame(encode_cbor(validated, {"maxByteLength": max_frame_length}))
        assert_complete_frame(frame, {"maxFrameLength": max_frame_length})
        return frame
    except ProtocolValidationError:
        raise
    except Exception as error:
        raise ProtocolValidationError(f"Unable to encode {kind} protocol message: {_bounded_error_message(error)}") from error


def encode_client_message(message: ClientMessage, options: dict[str, int] | None = None) -> bytes:
    return _encode_protocol_message(message, parse_client_message, "client", options)


def encode_server_message(message: ServerMessage, options: dict[str, int] | None = None) -> bytes:
    return _encode_protocol_message(message, parse_server_message, "server", options)


class ValidatedMessageDecoder:
    def __init__(self, kind: str, parse: Callable[[object], T], options: dict[str, int] | None = None) -> None:
        self._failed = False
        self._frames = FrameDecoder(options)
        self._kind = kind
        self._max_frame_length = (options or {}).get(
            "maxFrameLength", (options or {}).get("max_frame_length", DEFAULT_MAX_FRAME_LENGTH)
        )
        self._parse = parse

    def push(self, chunk: bytes | bytearray | memoryview) -> list[Any]:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            messages = []
            for frame in self._frames.push(chunk):
                messages.append(self._parse(decode_cbor(frame, {"maxByteLength": self._max_frame_length})))
            return messages
        except ProtocolValidationError:
            self._failed = True
            raise
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(f"Invalid {self._kind} protocol frame: {_bounded_error_message(error)}") from error

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(f"Invalid {self._kind} protocol framing: {_bounded_error_message(error)}") from error


class ClientMessageDecoder:
    def __init__(self, options: dict[str, int] | None = None) -> None:
        self._decoder = ValidatedMessageDecoder("client", parse_client_message, options)

    def push(self, chunk: bytes | bytearray | memoryview) -> list[ClientMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


class ServerMessageDecoder:
    def __init__(self, options: dict[str, int] | None = None) -> None:
        self._decoder = ValidatedMessageDecoder("server", parse_server_message, options)

    def push(self, chunk: bytes | bytearray | memoryview) -> list[ServerMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


def create_client_message_decoder(options: dict[str, int] | None = None) -> ClientMessageDecoder:
    return ClientMessageDecoder(options)


def create_server_message_decoder(options: dict[str, int] | None = None) -> ServerMessageDecoder:
    return ServerMessageDecoder(options)


def is_supported_protocol_version(version: object) -> bool:
    return isinstance(version, int) and not isinstance(version, bool) and version == PROTOCOL_VERSION
