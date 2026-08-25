from __future__ import annotations

import math
import struct

from .options import UINT32_BASE, CborError, ResolvedCborOptions, resolve_options


class CborReader:
    def __init__(self, data: bytes, options: ResolvedCborOptions) -> None:
        self.bytes = data
        self.offset = 0
        self.options = options

    def decode(self) -> object:
        value = self._read_item(0)
        if self.offset != len(self.bytes):
            raise CborError("CBOR payload contains trailing data")
        return value

    def _read_item(self, depth: int) -> object:
        if depth > self.options.max_depth:
            raise CborError(f"CBOR nesting depth exceeds configured limit of {self.options.max_depth}")
        initial = self._read_byte()
        major_type = initial >> 5
        additional = initial & 0x1F

        if major_type == 0:
            return self._read_argument(additional)
        if major_type == 1:
            value = -1 - self._read_argument(additional)
            if not (-(2**53) + 1 <= value <= 2**53 - 1):
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if major_type == 2:
            length = self._read_length(additional, "byte string", self.options.max_byte_length)
            return bytes(self._read_bytes(length))
        if major_type == 3:
            length = self._read_length(additional, "text string", self.options.max_byte_length)
            data = self._read_bytes(length)
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise CborError("CBOR text string contains invalid UTF-8") from error
        if major_type == 4:
            length = self._read_length(additional, "array", self.options.max_container_length)
            return [self._read_item(depth + 1) for _ in range(length)]
        if major_type == 5:
            length = self._read_length(additional, "map", self.options.max_container_length)
            result: dict[str, object] = {}
            keys: set[str] = set()
            for _ in range(length):
                key = self._read_item(depth + 1)
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                if key in keys:
                    raise CborError("CBOR map contains a duplicate key")
                keys.add(key)
                result[key] = self._read_item(depth + 1)
            return result
        if major_type == 6:
            raise CborError("CBOR tags are not supported")
        if major_type == 7:
            return self._read_simple(additional)
        raise CborError("Malformed CBOR major type")

    def _read_simple(self, additional: int) -> object:
        if additional == 20:
            return False
        if additional == 21:
            return True
        if additional == 22:
            return None
        if additional == 27:
            data = self._read_bytes(8)
            value = struct.unpack(">d", data)[0]
            if not math.isfinite(value):
                raise CborError("Decoded CBOR number must be finite")
            if float(value).is_integer() and not (-(2**53) + 1 <= int(value) <= 2**53 - 1):
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if additional == 31:
            raise CborError("CBOR break marker is not supported")
        raise CborError("Unsupported CBOR simple value or floating-point width")

    def _read_length(self, additional: int, kind: str, limit: int) -> int:
        if additional == 31:
            raise CborError(f"Indefinite-length CBOR {kind}s are not supported")
        length = self._read_argument(additional)
        if length > limit:
            raise CborError(f"CBOR {kind} length exceeds configured limit of {limit}")
        return length

    def _read_argument(self, additional: int) -> int:
        if additional < 24:
            return additional
        if additional == 24:
            return self._read_byte()
        if additional == 25:
            data = self._read_bytes(2)
            return data[0] * 0x100 + data[1]
        if additional == 26:
            data = self._read_bytes(4)
            return data[0] * 0x1_000_000 + data[1] * 0x1_0000 + data[2] * 0x100 + data[3]
        if additional == 27:
            high = self._read_argument(26)
            low = self._read_argument(26)
            if high > 0x1F_FFFF:
                raise CborError("Decoded CBOR integer or length is outside the safe range")
            return high * UINT32_BASE + low
        if additional == 31:
            raise CborError("Indefinite-length CBOR items are not supported")
        raise CborError("Malformed CBOR additional information")

    def _read_byte(self) -> int:
        if self.offset >= len(self.bytes):
            raise CborError("Truncated CBOR payload")
        value = self.bytes[self.offset]
        self.offset += 1
        return value

    def _read_bytes(self, length: int) -> bytes:
        if length > len(self.bytes) - self.offset:
            raise CborError("Truncated CBOR payload")
        value = self.bytes[self.offset : self.offset + length]
        self.offset += length
        return value


def decode_cbor(data: bytes | bytearray | memoryview, options: dict[str, int] | None = None) -> object:
    """Decodes exactly one item from the protocol's strict RFC 8949 subset."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("CBOR input must be a Uint8Array")
    payload = bytes(data)
    resolved = resolve_options(options)
    if len(payload) > resolved.max_byte_length:
        raise CborError(f"CBOR byte length exceeds configured limit of {resolved.max_byte_length}")
    return CborReader(payload, resolved).decode()
