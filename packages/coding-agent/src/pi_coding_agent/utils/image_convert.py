"""
Image conversion utilities — mirrors packages/coding-agent/src/utils/image-convert.ts

Converts images to PNG format (e.g. for terminal display via Kitty graphics protocol).
"""
from __future__ import annotations

import base64
import io


async def convert_to_png(
    base64_data: str,
    mime_type: str,
) -> dict[str, str] | None:
    """
    Convert image to PNG format for terminal display.
    Uses Pillow (replaces Photon WASM from TS).

    Returns {"data": base64_png, "mime_type": "image/png"} or None on failure.
    Mirrors convertToPng() in TypeScript.
    """
    if mime_type == "image/png":
        return {"data": base64_data, "mime_type": mime_type}

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        raw = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_data = base64.b64encode(buf.getvalue()).decode()
        return {"data": png_data, "mime_type": "image/png"}
    except Exception:
        return None
