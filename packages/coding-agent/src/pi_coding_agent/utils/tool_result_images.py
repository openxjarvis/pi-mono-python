"""
Normalize tool-result images — mirrors packages/coding-agent/src/utils/tool-result-images.ts
"""
from __future__ import annotations

from typing import Any

from .image_resize import resize_image


async def normalize_tool_result_images(
    content: list[Any],
    auto_resize_images: bool = True,
) -> list[Any]:
    if not any((item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "image" for item in content):
        return content
    if not auto_resize_images:
        return content

    normalized: list[Any] = []
    changed = False
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type != "image":
            normalized.append(block)
            continue
        data = block.get("data") if isinstance(block, dict) else getattr(block, "data", "")
        mime = block.get("mimeType") if isinstance(block, dict) else getattr(block, "mime_type", "image/png")
        try:
            processed = await resize_image(data, mime or "image/png")
        except Exception:
            normalized.append(block)
            continue
        if processed.data == data and processed.mime_type == mime:
            normalized.append(block)
            continue
        normalized.append({"type": "image", "data": processed.data, "mimeType": processed.mime_type})
        changed = True
    return normalized if changed else content
