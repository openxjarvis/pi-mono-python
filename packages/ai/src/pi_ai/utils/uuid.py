"""
Time-ordered UUIDv7.
Mirrors packages/ai/src/utils/uuid.ts
"""
from __future__ import annotations

import os
import time

_last_timestamp = float("-inf")
_sequence = 0


def uuidv7() -> str:
    global _last_timestamp, _sequence
    random_bytes = os.urandom(16)
    timestamp = int(time.time() * 1000)

    if timestamp > _last_timestamp:
        _sequence = (
            random_bytes[6] * 0x1000000
            + random_bytes[7] * 0x10000
            + random_bytes[8] * 0x100
            + random_bytes[9]
        )
        _last_timestamp = timestamp
    else:
        _sequence = (_sequence + 1) & 0xFFFFFFFF
        if _sequence == 0:
            _last_timestamp += 1

    ts = int(_last_timestamp)
    seq = _sequence
    bytes_out = bytearray(16)
    bytes_out[0] = (ts // 0x10000000000) & 0xFF
    bytes_out[1] = (ts // 0x100000000) & 0xFF
    bytes_out[2] = (ts // 0x1000000) & 0xFF
    bytes_out[3] = (ts // 0x10000) & 0xFF
    bytes_out[4] = (ts // 0x100) & 0xFF
    bytes_out[5] = ts & 0xFF
    bytes_out[6] = 0x70 | ((seq >> 28) & 0x0F)
    bytes_out[7] = (seq >> 20) & 0xFF
    bytes_out[8] = 0x80 | ((seq >> 14) & 0x3F)
    bytes_out[9] = (seq >> 6) & 0xFF
    bytes_out[10] = ((seq & 0x3F) << 2) | (random_bytes[10] & 0x03)
    bytes_out[11] = random_bytes[11]
    bytes_out[12] = random_bytes[12]
    bytes_out[13] = random_bytes[13]
    bytes_out[14] = random_bytes[14]
    bytes_out[15] = random_bytes[15]

    hex_bytes = [f"{b:02x}" for b in bytes_out]
    return (
        f"{''.join(hex_bytes[0:4])}-{''.join(hex_bytes[4:6])}-"
        f"{''.join(hex_bytes[6:8])}-{''.join(hex_bytes[8:10])}-{''.join(hex_bytes[10:16])}"
    )
