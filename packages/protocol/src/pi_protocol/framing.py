from __future__ import annotations

FRAME_HEADER_LENGTH = 4
MAX_UINT32 = 0xFFFF_FFFF
PAYLOAD_BLOCK_SIZE = 64 * 1024
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024


class FrameDecoderOptions(dict):
    pass


class FrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "FrameError"


def resolve_max_frame_length(options: dict[str, int] | None = None) -> int:
    value = DEFAULT_MAX_FRAME_LENGTH
    if options:
        value = options.get("maxFrameLength", options.get("max_frame_length", DEFAULT_MAX_FRAME_LENGTH))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > MAX_UINT32:
        raise ValueError(f"maxFrameLength must be an integer between 0 and {MAX_UINT32}")
    return value


def encode_frame(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Frame payload must be a Uint8Array")
    data = bytes(payload)
    if len(data) > MAX_UINT32:
        raise ValueError("Frame payload exceeds the unsigned 32-bit length limit")
    length = len(data)
    header = bytes([(length >> 24) & 0xFF, (length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF])
    return header + data


def assert_complete_frame(frame: bytes | bytearray | memoryview, options: dict[str, int] | None = None) -> None:
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("Frame must be a Uint8Array")
    data = bytes(frame)
    if len(data) < FRAME_HEADER_LENGTH:
        raise FrameError("Frame does not contain a complete length prefix")
    length = data[0] * 0x1_000_000 + data[1] * 0x1_0000 + data[2] * 0x100 + data[3]
    max_frame_length = resolve_max_frame_length(options)
    if length > max_frame_length:
        raise FrameError(f"Frame length {length} exceeds configured limit of {max_frame_length}")
    if len(data) != FRAME_HEADER_LENGTH + length:
        raise FrameError("Frame must contain exactly one complete payload")


class FrameDecoder:
    def __init__(self, options: dict[str, int] | None = None) -> None:
        self._header = bytearray(FRAME_HEADER_LENGTH)
        self._header_length = 0
        self._max_frame_length = resolve_max_frame_length(options)
        self._payload_blocks: list[bytearray] = []
        self._current_payload_block: bytearray | None = None
        self._current_payload_block_length = 0
        self._expected_payload_length: int | None = None
        self._payload_length = 0
        self._state = "open"

    def push(self, chunk: bytes | bytearray | memoryview) -> list[bytes]:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("Frame chunk must be a Uint8Array")
        data = bytes(chunk)
        frames: list[bytes] = []
        offset = 0
        while offset < len(data):
            if self._expected_payload_length is None:
                header_bytes = min(FRAME_HEADER_LENGTH - self._header_length, len(data) - offset)
                self._header[self._header_length : self._header_length + header_bytes] = data[offset : offset + header_bytes]
                self._header_length += header_bytes
                offset += header_bytes
                if self._header_length < FRAME_HEADER_LENGTH:
                    continue
                frame_length = (
                    self._header[0] * 0x1_000_000 + self._header[1] * 0x1_0000 + self._header[2] * 0x100 + self._header[3]
                )
                self._header_length = 0
                if frame_length > self._max_frame_length:
                    self._fail(f"Frame length {frame_length} exceeds configured limit of {self._max_frame_length}")
                if frame_length == 0:
                    frames.append(b"")
                    continue
                self._expected_payload_length = frame_length
                self._payload_blocks = []
                self._current_payload_block = None
                self._current_payload_block_length = 0
                self._payload_length = 0

            expected = self._expected_payload_length
            if expected is None:
                continue
            while offset < len(data) and self._payload_length < expected:
                block = self._current_payload_block
                if block is None or self._current_payload_block_length == len(block):
                    block = bytearray(min(PAYLOAD_BLOCK_SIZE, expected - self._payload_length))
                    self._payload_blocks.append(block)
                    self._current_payload_block = block
                    self._current_payload_block_length = 0
                payload_bytes = min(len(block) - self._current_payload_block_length, len(data) - offset)
                block[self._current_payload_block_length : self._current_payload_block_length + payload_bytes] = data[
                    offset : offset + payload_bytes
                ]
                self._current_payload_block_length += payload_bytes
                self._payload_length += payload_bytes
                offset += payload_bytes
            if self._payload_length == expected:
                if len(self._payload_blocks) == 1:
                    frames.append(bytes(self._payload_blocks[0]))
                else:
                    payload = bytearray(expected)
                    cursor = 0
                    for payload_block in self._payload_blocks:
                        payload[cursor : cursor + len(payload_block)] = payload_block
                        cursor += len(payload_block)
                    frames.append(bytes(payload))
                self._payload_blocks = []
                self._current_payload_block = None
                self._current_payload_block_length = 0
                self._expected_payload_length = None
                self._payload_length = 0
        return frames

    def end(self) -> None:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if self._header_length != 0 or self._expected_payload_length is not None:
            self._fail("Truncated frame at end of stream")
        self._state = "ended"

    def _fail(self, message: str) -> None:
        self._state = "failed"
        self._header_length = 0
        self._payload_blocks = []
        self._current_payload_block = None
        self._current_payload_block_length = 0
        self._expected_payload_length = None
        self._payload_length = 0
        raise FrameError(message)
