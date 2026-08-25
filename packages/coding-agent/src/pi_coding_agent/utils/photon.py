"""
Python equivalent of utils/photon.ts.

TypeScript wraps ``@silvia-odwyer/photon-node`` (Rust/WASM). Python uses
Pillow for the same resize/encode surface so callers do not depend on WASM.
"""
from __future__ import annotations

import io
from typing import Any


class PhotonImage:
    """Pillow-backed stand-in for photon-node PhotonImage."""

    def __init__(self, image: Any) -> None:
        self._image = image

    @classmethod
    def new_from_byteslice(cls, data: bytes | bytearray | memoryview) -> "PhotonImage":
        from PIL import Image

        image = Image.open(io.BytesIO(bytes(data)))
        image.load()
        return cls(image)

    def get_width(self) -> int:
        return int(self._image.size[0])

    def get_height(self) -> int:
        return int(self._image.size[1])

    def get_bytes(self) -> bytes:
        buf = io.BytesIO()
        self._image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def get_bytes_jpeg(self, quality: int) -> bytes:
        buf = io.BytesIO()
        self._image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    def free(self) -> None:
        self._image = None


class SamplingFilter:
    Lanczos3 = "lanczos3"


class _PhotonModule:
    PhotonImage = PhotonImage
    SamplingFilter = SamplingFilter

    def resize(self, image: PhotonImage, width: int, height: int, _filter: str | None = None) -> PhotonImage:
        from PIL import Image

        resized = image._image.resize((max(1, width), max(1, height)), Image.LANCZOS)
        return PhotonImage(resized)


_photon_module: _PhotonModule | None = None
_load_failed = False


async def load_photon() -> _PhotonModule | None:
    """Load the Pillow-backed photon equivalent, or None if Pillow is missing."""
    global _photon_module, _load_failed
    if _photon_module is not None:
        return _photon_module
    if _load_failed:
        return None
    try:
        import PIL  # noqa: F401
    except ImportError:
        _load_failed = True
        return None
    _photon_module = _PhotonModule()
    return _photon_module
