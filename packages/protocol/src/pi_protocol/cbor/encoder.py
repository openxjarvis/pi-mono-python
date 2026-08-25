from __future__ import annotations

import math
import struct
from typing import Any

from .options import MAX_UINT32, UINT32_BASE, CborError, ResolvedCborOptions, resolve_options


class CborWriter:
    def __init__(self, max_byte_length: int) -> None:
        self.max_byte_length = max_byte_length
        self.buffer = bytearray(min(256, max_byte_length))
        self.offset = 0

    def write_byte(self, value: int) -> None:
        self._ensure_capacity(1)
        self.buffer[self.offset] = value & 0xFF
        self.offset += 1

    def write_bytes(self, data: bytes | bytearray | memoryview) -> None:
        self._ensure_capacity(len(data))
        self.buffer[self.offset : self.offset + len(data)] = data
        self.offset += len(data)

    def write_uint16(self, value: int) -> None:
        self._ensure_capacity(2)
        self.buffer[self.offset] = (value >> 8) & 0xFF
        self.buffer[self.offset + 1] = value & 0xFF
        self.offset += 2

    def write_uint32(self, value: int) -> None:
        self._ensure_capacity(4)
        self.buffer[self.offset] = (value >> 24) & 0xFF
        self.buffer[self.offset + 1] = (value >> 16) & 0xFF
        self.buffer[self.offset + 2] = (value >> 8) & 0xFF
        self.buffer[self.offset + 3] = value & 0xFF
        self.offset += 4

    def write_uint64(self, value: int) -> None:
        high = value // UINT32_BASE
        low = value - high * UINT32_BASE
        self.write_uint32(high)
        self.write_uint32(low)

    def write_float64(self, value: float) -> None:
        self._ensure_capacity(9)
        self.buffer[self.offset] = 0xFB
        self.buffer[self.offset + 1 : self.offset + 9] = struct.pack(">d", value)
        self.offset += 9

    def finish(self) -> bytes:
        return bytes(self.buffer[: self.offset])

    def _ensure_capacity(self, additional_bytes: int) -> None:
        required = self.offset + additional_bytes
        if required > self.max_byte_length:
            raise CborError(f"CBOR byte length exceeds configured limit of {self.max_byte_length}")
        if required <= len(self.buffer):
            return
        capacity = max(1, len(self.buffer))
        while capacity < required:
            capacity = min(self.max_byte_length, max(required, capacity * 2))
        expanded = bytearray(capacity)
        expanded[: self.offset] = self.buffer[: self.offset]
        self.buffer = expanded


def _write_argument(writer: CborWriter, major_type: int, value: int) -> None:
    prefix = major_type << 5
    if value < 24:
        writer.write_byte(prefix | value)
    elif value <= 0xFF:
        writer.write_byte(prefix | 24)
        writer.write_byte(value)
    elif value <= 0xFFFF:
        writer.write_byte(prefix | 25)
        writer.write_uint16(value)
    elif value <= MAX_UINT32:
        writer.write_byte(prefix | 26)
        writer.write_uint32(value)
    else:
        writer.write_byte(prefix | 27)
        writer.write_uint64(value)


def _is_plain_object(value: object) -> bool:
    return isinstance(value, dict)


def _encode_text(writer: CborWriter, value: str, options: ResolvedCborOptions) -> None:
    data = value.encode("utf-8")
    if len(data) > options.max_byte_length:
        raise CborError(f"CBOR text string length exceeds configured limit of {options.max_byte_length}")
    if data.decode("utf-8") != value:
        raise CborError("CBOR text strings must contain valid Unicode scalar values")
    _write_argument(writer, 3, len(data))
    writer.write_bytes(data)


def _is_safe_integer(value: int) -> bool:
    return -(2**53) + 1 <= value <= 2**53 - 1


def _encode_value(
    writer: CborWriter,
    value: Any,
    options: ResolvedCborOptions,
    depth: int,
    ancestors: set[int],
) -> None:
    if depth > options.max_depth:
        raise CborError(f"CBOR nesting depth exceeds configured limit of {options.max_depth}")

    if value is None:
        writer.write_byte(0xF6)
        return
    if isinstance(value, bool):
        writer.write_byte(0xF5 if value else 0xF4)
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if not _is_safe_integer(value):
            raise CborError("CBOR integers must be safe JavaScript integers")
        if value >= 0:
            _write_argument(writer, 0, value)
        else:
            _write_argument(writer, 1, -1 - value)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CborError("CBOR numbers must be finite")
        if value.is_integer() and not math.copysign(1.0, value) < 0:
            integer = int(value)
            if not _is_safe_integer(integer):
                raise CborError("CBOR integers must be safe JavaScript integers")
            if integer >= 0:
                _write_argument(writer, 0, integer)
            else:
                _write_argument(writer, 1, -1 - integer)
            return
        writer.write_float64(value)
        return
    if isinstance(value, str):
        _encode_text(writer, value, options)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if len(data) > options.max_byte_length:
            raise CborError(f"CBOR byte string length exceeds configured limit of {options.max_byte_length}")
        _write_argument(writer, 2, len(data))
        writer.write_bytes(data)
        return
    if isinstance(value, list):
        ident = id(value)
        if ident in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > options.max_container_length:
            raise CborError(f"CBOR array length exceeds configured limit of {options.max_container_length}")
        ancestors.add(ident)
        try:
            _write_argument(writer, 4, len(value))
            for item in value:
                if item is None and False:
                    pass
                _encode_value(writer, item, options, depth + 1, ancestors)
        finally:
            ancestors.discard(ident)
        return
    if _is_plain_object(value):
        ident = id(value)
        if ident in ancestors:
            raise CborError("CBOR values must not contain cycles")
        entries = [(key, entry) for key, entry in value.items() if entry is not None]
        if any(not isinstance(key, str) for key, _ in entries):
            raise CborError("CBOR map keys must be strings")
        if len(entries) > options.max_container_length:
            raise CborError(f"CBOR map length exceeds configured limit of {options.max_container_length}")
        ancestors.add(ident)
        try:
            _write_argument(writer, 5, len(entries))
            for key, entry_value in entries:
                _encode_text(writer, key, options)
                _encode_value(writer, entry_value, options, depth + 1, ancestors)
        finally:
            ancestors.discard(ident)
        return

    raise CborError(f"Unsupported CBOR value type: {type(value).__name__}")


def encode_cbor(value: object, options: dict[str, int] | None = None) -> bytes:
    """Encodes the protocol's strict, definite-length RFC 8949 subset."""
    resolved = resolve_options(options)
    writer = CborWriter(resolved.max_byte_length)
    _encode_value(writer, value, resolved, 0, set())
    return writer.finish()
