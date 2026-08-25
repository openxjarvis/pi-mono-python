"""
Clipboard image helpers — mirrors utils/clipboard-image.ts.

Re-exports the portable implementation in clipboard.py and adds the
native-module + Photon conversion path used by InteractiveMode.
"""
from __future__ import annotations

from .clipboard import (
    ClipboardImage,
    SUPPORTED_IMAGE_MIME_TYPES,
    copy_image_to_clipboard,
    extension_for_image_mime_type,
    read_clipboard_image as _read_clipboard_image,
)
from .clipboard_native import clipboard
from .photon import load_photon


def is_wayland_session(env: dict[str, str] | None = None) -> bool:
    source = env if env is not None else __import__("os").environ
    return bool(source.get("WAYLAND_DISPLAY") or source.get("XDG_SESSION_TYPE") == "wayland")


async def _convert_to_png(data: bytes) -> bytes | None:
    photon = await load_photon()
    if photon is None:
        return None
    try:
        image = photon.PhotonImage.new_from_byteslice(data)
        try:
            return image.get_bytes()
        finally:
            image.free()
    except Exception:
        return None


async def _read_via_native() -> ClipboardImage | None:
    if clipboard is None or not clipboard.has_image():
        return None
    raw = await clipboard.get_image_binary()
    if not raw:
        return None
    return ClipboardImage(data=bytes(raw), mime_type="image/png")


async def read_clipboard_image() -> ClipboardImage | None:
    image = _read_clipboard_image()
    if image is None:
        image = await _read_via_native()
    if image is None:
        return None
    mime = image.mime_type.split(";")[0].strip().lower()
    if mime not in SUPPORTED_IMAGE_MIME_TYPES:
        converted = await _convert_to_png(image.data)
        if converted is None:
            return None
        return ClipboardImage(data=converted, mime_type="image/png")
    return image


__all__ = [
    "ClipboardImage",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "copy_image_to_clipboard",
    "extension_for_image_mime_type",
    "is_wayland_session",
    "read_clipboard_image",
]
