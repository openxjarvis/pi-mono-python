"""
Off-thread image resize — mirrors utils/image-resize-worker.ts.

TypeScript uses ``worker_threads``. Python uses a thread so Pillow work
does not block the asyncio event loop.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .image_resize_core import ImageResizeOptions, ResizedImage, resize_image_in_process

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-image-resize")


async def resize_image_in_worker(
    input_bytes: bytes | bytearray | memoryview,
    mime_type: str,
    options: ImageResizeOptions | None = None,
) -> ResizedImage | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _run_sync,
        bytes(input_bytes),
        mime_type,
        options,
    )


def _run_sync(
    input_bytes: bytes,
    mime_type: str,
    options: ImageResizeOptions | None,
) -> ResizedImage | None:
    try:
        return asyncio.run(resize_image_in_process(input_bytes, mime_type, options))
    except Exception:
        return None


def handle_worker_request(message: dict[str, Any]) -> dict[str, Any]:
    """Same request/response shape as the Node worker."""
    try:
        input_bytes = message.get("inputBytes") or message.get("input_bytes")
        mime_type = message.get("mimeType") or message.get("mime_type")
        if not isinstance(input_bytes, (bytes, bytearray, memoryview)) or not isinstance(mime_type, str):
            raise ValueError("Invalid image resize worker request")
        result = _run_sync(bytes(input_bytes), mime_type, message.get("options"))
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}
