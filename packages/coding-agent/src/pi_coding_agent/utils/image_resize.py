"""
Image resize utilities — mirrors packages/coding-agent/src/utils/image-resize.ts

Uses Pillow instead of Photon (WASM) for image processing.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

# 4.5MB — headroom below Anthropic's 5MB limit
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
    data: str          # base64
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


def _encode_png(img: "Image") -> tuple[bytes, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


def _encode_jpeg(img: "Image", quality: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _pick_smaller(
    a: tuple[bytes, str],
    b: tuple[bytes, str],
) -> tuple[bytes, str]:
    return a if len(a[0]) <= len(b[0]) else b


async def resize_image(
    data: str,
    mime_type: str,
    options: ImageResizeOptions | None = None,
) -> ResizedImage:
    """
    Resize an image to fit within specified max dimensions and file size.
    Uses Pillow (replaces Photon WASM from TS).

    Strategy mirrors TypeScript:
    1. Resize to maxWidth/maxHeight
    2. Try PNG and JPEG — pick smaller
    3. If still too large, try JPEG with decreasing quality
    4. If still too large, progressively reduce dimensions

    Returns original if Pillow is unavailable or image already fits.
    """
    opts = options or ImageResizeOptions()

    try:
        from PIL import Image
    except ImportError:
        return ResizedImage(
            data=data,
            mime_type=mime_type,
            original_width=0,
            original_height=0,
            width=0,
            height=0,
            was_resized=False,
        )

    try:
        raw = base64.b64decode(data)
        img = Image.open(io.BytesIO(raw))
        original_width, original_height = img.size

        # Check if already fits
        if (
            original_width <= opts.max_width
            and original_height <= opts.max_height
            and len(raw) <= opts.max_bytes
        ):
            return ResizedImage(
                data=data,
                mime_type=mime_type,
                original_width=original_width,
                original_height=original_height,
                width=original_width,
                height=original_height,
                was_resized=False,
            )

        # Calculate initial target dimensions
        target_w, target_h = original_width, original_height
        if target_w > opts.max_width:
            target_h = round(target_h * opts.max_width / target_w)
            target_w = opts.max_width
        if target_h > opts.max_height:
            target_w = round(target_w * opts.max_height / target_h)
            target_h = opts.max_height

        quality_steps = [85, 70, 55, 40]
        scale_steps = [1.0, 0.75, 0.5, 0.35, 0.25]

        def try_both_formats(w: int, h: int, quality: int) -> tuple[bytes, str]:
            resized = img.resize((w, h), Image.LANCZOS)
            png = _encode_png(resized)
            jpeg = _encode_jpeg(resized, quality)
            return _pick_smaller(png, jpeg)

        # First attempt at target size
        best_bytes, best_mime = try_both_formats(target_w, target_h, opts.jpeg_quality)
        final_w, final_h = target_w, target_h

        if len(best_bytes) <= opts.max_bytes:
            return ResizedImage(
                data=base64.b64encode(best_bytes).decode(),
                mime_type=best_mime,
                original_width=original_width,
                original_height=original_height,
                width=final_w,
                height=final_h,
                was_resized=True,
            )

        # Try decreasing JPEG quality at target size
        for quality in quality_steps:
            best_bytes, best_mime = try_both_formats(target_w, target_h, quality)
            if len(best_bytes) <= opts.max_bytes:
                return ResizedImage(
                    data=base64.b64encode(best_bytes).decode(),
                    mime_type=best_mime,
                    original_width=original_width,
                    original_height=original_height,
                    width=final_w,
                    height=final_h,
                    was_resized=True,
                )

        # Progressively reduce dimensions
        for scale in scale_steps:
            sw = round(target_w * scale)
            sh = round(target_h * scale)
            if sw < 100 or sh < 100:
                break
            final_w, final_h = sw, sh
            for quality in quality_steps:
                best_bytes, best_mime = try_both_formats(sw, sh, quality)
                if len(best_bytes) <= opts.max_bytes:
                    return ResizedImage(
                        data=base64.b64encode(best_bytes).decode(),
                        mime_type=best_mime,
                        original_width=original_width,
                        original_height=original_height,
                        width=final_w,
                        height=final_h,
                        was_resized=True,
                    )

        # Last resort — return smallest produced
        return ResizedImage(
            data=base64.b64encode(best_bytes).decode(),
            mime_type=best_mime,
            original_width=original_width,
            original_height=original_height,
            width=final_w,
            height=final_h,
            was_resized=True,
        )

    except Exception:
        return ResizedImage(
            data=data,
            mime_type=mime_type,
            original_width=0,
            original_height=0,
            width=0,
            height=0,
            was_resized=False,
        )


def format_dimension_note(result: ResizedImage) -> str | None:
    """
    Format a dimension note for resized images.
    Mirrors formatDimensionNote() in TypeScript.
    """
    if not result.was_resized:
        return None
    scale = result.original_width / result.width if result.width > 0 else 1.0
    return (
        f"[Image: original {result.original_width}x{result.original_height}, "
        f"displayed at {result.width}x{result.height}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )
