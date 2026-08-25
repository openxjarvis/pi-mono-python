"""
In-process image resize — mirrors utils/image-resize-core.ts.

Uses the Pillow photon equivalent instead of photon-node WASM.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

from .photon import load_photon

DEFAULT_MAX_BYTES = int(4.5 * 1024 * 1024)
DEFAULT_MAX_WIDTH = 2000
DEFAULT_MAX_HEIGHT = 2000
DEFAULT_JPEG_QUALITY = 80


@dataclass
class ImageResizeOptions:
    max_width: int = DEFAULT_MAX_WIDTH
    max_height: int = DEFAULT_MAX_HEIGHT
    max_bytes: int = DEFAULT_MAX_BYTES
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


@dataclass
class ResizedImage:
    data: str
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


def _encode_candidate(buffer: bytes, mime_type: str) -> dict[str, object]:
    data = base64.b64encode(buffer).decode("ascii")
    return {"data": data, "encoded_size": len(data.encode("utf-8")), "mime_type": mime_type}


async def resize_image_in_process(
    input_bytes: bytes | bytearray | memoryview,
    mime_type: str,
    options: ImageResizeOptions | None = None,
) -> ResizedImage | None:
    opts = options or ImageResizeOptions()
    raw = bytes(input_bytes)
    input_base64_size = ((len(raw) + 2) // 3) * 4
    photon = await load_photon()
    if photon is None:
        return None

    image = None
    try:
        raw_image = photon.PhotonImage.new_from_byteslice(raw)
        image = raw_image
        original_width = image.get_width()
        original_height = image.get_height()
        fmt = (mime_type.split("/")[1] if "/" in mime_type else "png") or "png"

        if original_width <= opts.max_width and original_height <= opts.max_height and input_base64_size < opts.max_bytes:
            return ResizedImage(
                data=base64.b64encode(raw).decode("ascii"),
                mime_type=mime_type or f"image/{fmt}",
                original_width=original_width,
                original_height=original_height,
                width=original_width,
                height=original_height,
                was_resized=False,
            )

        target_width, target_height = original_width, original_height
        if target_width > opts.max_width:
            target_height = round(target_height * opts.max_width / target_width)
            target_width = opts.max_width
        if target_height > opts.max_height:
            target_width = round(target_width * opts.max_height / target_height)
            target_height = opts.max_height

        qualities = list(dict.fromkeys([opts.jpeg_quality, 85, 70, 55, 40]))
        current_width, current_height = target_width, target_height
        while True:
            resized = photon.resize(image, current_width, current_height, photon.SamplingFilter.Lanczos3)
            try:
                candidates = [_encode_candidate(resized.get_bytes(), "image/png")]
                for quality in qualities:
                    candidates.append(_encode_candidate(resized.get_bytes_jpeg(quality), "image/jpeg"))
            finally:
                resized.free()
            for candidate in candidates:
                if int(candidate["encoded_size"]) < opts.max_bytes:
                    return ResizedImage(
                        data=str(candidate["data"]),
                        mime_type=str(candidate["mime_type"]),
                        original_width=original_width,
                        original_height=original_height,
                        width=current_width,
                        height=current_height,
                        was_resized=True,
                    )
            if current_width == 1 and current_height == 1:
                break
            next_width = 1 if current_width == 1 else max(1, current_width * 3 // 4)
            next_height = 1 if current_height == 1 else max(1, current_height * 3 // 4)
            if next_width == current_width and next_height == current_height:
                break
            current_width, current_height = next_width, next_height
        return None
    except Exception:
        return None
    finally:
        if image is not None:
            image.free()
