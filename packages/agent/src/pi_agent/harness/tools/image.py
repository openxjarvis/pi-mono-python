"""Image helpers. Mirrors packages/agent/src/harness/tools/image.ts"""
from __future__ import annotations

import base64

_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"BM": "image/bmp",
    b"RIFF": "image/webp",
}


def detect_supported_image_mime_type(data: bytes) -> str | None:
    for magic, mime in _MAGIC.items():
        if data.startswith(magic):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return None


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
