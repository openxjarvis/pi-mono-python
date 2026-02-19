"""
Terminal image protocols — mirrors packages/tui/src/terminal-image.ts

Supports Kitty Graphics Protocol and iTerm2 inline images.
Detects terminal capabilities from environment variables.
"""
from __future__ import annotations

import base64
import os
import random
import struct

# ─────────────────────────────────────────────────────────────────────────────
# Types / data classes
# ─────────────────────────────────────────────────────────────────────────────

ImageProtocol = str | None  # "kitty" | "iterm2" | None


class TerminalCapabilities:
    __slots__ = ("images", "true_color", "hyperlinks")

    def __init__(
        self,
        images: ImageProtocol = None,
        true_color: bool = False,
        hyperlinks: bool = False,
    ) -> None:
        self.images = images
        self.true_color = true_color
        self.hyperlinks = hyperlinks


class CellDimensions:
    __slots__ = ("width_px", "height_px")

    def __init__(self, width_px: int = 9, height_px: int = 18) -> None:
        self.width_px = width_px
        self.height_px = height_px


class ImageDimensions:
    __slots__ = ("width_px", "height_px")

    def __init__(self, width_px: int, height_px: int) -> None:
        self.width_px = width_px
        self.height_px = height_px


class ImageRenderOptions:
    __slots__ = ("max_width_cells", "max_height_cells", "preserve_aspect_ratio", "image_id")

    def __init__(
        self,
        max_width_cells: int | None = None,
        max_height_cells: int | None = None,
        preserve_aspect_ratio: bool = True,
        image_id: int | None = None,
    ) -> None:
        self.max_width_cells = max_width_cells
        self.max_height_cells = max_height_cells
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.image_id = image_id


# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

_cached_capabilities: TerminalCapabilities | None = None
_cell_dimensions = CellDimensions(width_px=9, height_px=18)


def get_cell_dimensions() -> CellDimensions:
    return _cell_dimensions


def set_cell_dimensions(dims: CellDimensions) -> None:
    global _cell_dimensions
    _cell_dimensions = dims


def reset_capabilities_cache() -> None:
    global _cached_capabilities
    _cached_capabilities = None


# ─────────────────────────────────────────────────────────────────────────────
# Capability detection — mirrors detectCapabilities() in terminal-image.ts
# ─────────────────────────────────────────────────────────────────────────────

def detect_capabilities() -> TerminalCapabilities:
    term_program = (os.environ.get("TERM_PROGRAM") or "").lower()
    term = (os.environ.get("TERM") or "").lower()
    color_term = (os.environ.get("COLORTERM") or "").lower()

    if os.environ.get("KITTY_WINDOW_ID") or term_program == "kitty":
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    if term_program == "ghostty" or "ghostty" in term or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    if os.environ.get("WEZTERM_PANE") or term_program == "wezterm":
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    if os.environ.get("ITERM_SESSION_ID") or term_program == "iterm.app":
        return TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True)

    if term_program == "vscode":
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=True)

    if term_program == "alacritty":
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=True)

    true_color = color_term in ("truecolor", "24bit")
    return TerminalCapabilities(images=None, true_color=true_color, hyperlinks=True)


def get_capabilities() -> TerminalCapabilities:
    global _cached_capabilities
    if _cached_capabilities is None:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


# ─────────────────────────────────────────────────────────────────────────────
# Image line detection
# ─────────────────────────────────────────────────────────────────────────────

_KITTY_PREFIX = "\x1b_G"
_ITERM2_PREFIX = "\x1b]1337;File="


def is_image_line(line: str) -> bool:
    """Check if a rendered line contains an image sequence."""
    if line.startswith(_KITTY_PREFIX) or line.startswith(_ITERM2_PREFIX):
        return True
    return _KITTY_PREFIX in line or _ITERM2_PREFIX in line


# ─────────────────────────────────────────────────────────────────────────────
# Image ID allocation
# ─────────────────────────────────────────────────────────────────────────────

def allocate_image_id() -> int:
    """Generate a random Kitty graphics image ID in range [1, 0xffffffff]."""
    return random.randint(1, 0xffffffff)


# ─────────────────────────────────────────────────────────────────────────────
# Kitty protocol encoding — mirrors encodeKitty() in terminal-image.ts
# ─────────────────────────────────────────────────────────────────────────────

def encode_kitty(
    base64_data: str,
    columns: int | None = None,
    rows: int | None = None,
    image_id: int | None = None,
) -> str:
    """
    Encode image data as a Kitty Graphics Protocol sequence.
    Mirrors encodeKitty() in terminal-image.ts.
    """
    CHUNK_SIZE = 4096

    params = ["a=T", "f=100", "q=2"]
    if columns:
        params.append(f"c={columns}")
    if rows:
        params.append(f"r={rows}")
    if image_id:
        params.append(f"i={image_id}")

    if len(base64_data) <= CHUNK_SIZE:
        return f"\x1b_G{','.join(params)};{base64_data}\x1b\\"

    chunks: list[str] = []
    offset = 0
    is_first = True
    p = ",".join(params)

    while offset < len(base64_data):
        chunk = base64_data[offset:offset + CHUNK_SIZE]
        is_last = offset + CHUNK_SIZE >= len(base64_data)

        if is_first:
            chunks.append(f"\x1b_G{p},m=1;{chunk}\x1b\\")
            is_first = False
        elif is_last:
            chunks.append(f"\x1b_Gm=0;{chunk}\x1b\\")
        else:
            chunks.append(f"\x1b_Gm=1;{chunk}\x1b\\")

        offset += CHUNK_SIZE

    return "".join(chunks)


def delete_kitty_image(image_id: int) -> str:
    """Delete a Kitty graphics image by ID."""
    return f"\x1b_Ga=d,d=I,i={image_id}\x1b\\"


def delete_all_kitty_images() -> str:
    """Delete all visible Kitty graphics images."""
    return "\x1b_Ga=d,d=A\x1b\\"


# ─────────────────────────────────────────────────────────────────────────────
# iTerm2 protocol encoding — mirrors encodeITerm2() in terminal-image.ts
# ─────────────────────────────────────────────────────────────────────────────

def encode_iterm2(
    base64_data: str,
    width: int | str | None = None,
    height: int | str | None = None,
    name: str | None = None,
    preserve_aspect_ratio: bool = True,
    inline: bool = True,
) -> str:
    """
    Encode image data as an iTerm2 inline image sequence.
    Mirrors encodeITerm2() in terminal-image.ts.
    """
    params = [f"inline={1 if inline else 0}"]
    if width is not None:
        params.append(f"width={width}")
    if height is not None:
        params.append(f"height={height}")
    if name:
        name_b64 = base64.b64encode(name.encode()).decode()
        params.append(f"name={name_b64}")
    if not preserve_aspect_ratio:
        params.append("preserveAspectRatio=0")

    return f"\x1b]1337;File={';'.join(params)}:{base64_data}\x07"


# ─────────────────────────────────────────────────────────────────────────────
# Image dimension reading — mirrors getPngDimensions etc.
# ─────────────────────────────────────────────────────────────────────────────

def get_png_dimensions(base64_data: str) -> ImageDimensions | None:
    """Read PNG width/height from base64-encoded data."""
    try:
        buf = base64.b64decode(base64_data)
        if len(buf) < 24:
            return None
        if buf[:4] != b"\x89PNG":
            return None
        width = struct.unpack(">I", buf[16:20])[0]
        height = struct.unpack(">I", buf[20:24])[0]
        return ImageDimensions(width, height)
    except Exception:
        return None


def get_jpeg_dimensions(base64_data: str) -> ImageDimensions | None:
    """Read JPEG width/height from base64-encoded data."""
    try:
        buf = base64.b64decode(base64_data)
        if len(buf) < 2:
            return None
        if buf[0] != 0xFF or buf[1] != 0xD8:
            return None
        offset = 2
        while offset < len(buf) - 9:
            if buf[offset] != 0xFF:
                offset += 1
                continue
            marker = buf[offset + 1]
            if 0xC0 <= marker <= 0xC2:
                height = struct.unpack(">H", buf[offset + 5:offset + 7])[0]
                width = struct.unpack(">H", buf[offset + 7:offset + 9])[0]
                return ImageDimensions(width, height)
            if offset + 3 >= len(buf):
                return None
            length = struct.unpack(">H", buf[offset + 2:offset + 4])[0]
            if length < 2:
                return None
            offset += 2 + length
        return None
    except Exception:
        return None


def get_gif_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buf = base64.b64decode(base64_data)
        if len(buf) < 10:
            return None
        sig = buf[:6]
        if sig not in (b"GIF87a", b"GIF89a"):
            return None
        width = struct.unpack("<H", buf[6:8])[0]
        height = struct.unpack("<H", buf[8:10])[0]
        return ImageDimensions(width, height)
    except Exception:
        return None


def get_webp_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buf = base64.b64decode(base64_data)
        if len(buf) < 30:
            return None
        if buf[:4] != b"RIFF" or buf[8:12] != b"WEBP":
            return None
        chunk = buf[12:16]
        if chunk == b"VP8 ":
            if len(buf) < 30:
                return None
            width = struct.unpack("<H", buf[26:28])[0] & 0x3FFF
            height = struct.unpack("<H", buf[28:30])[0] & 0x3FFF
            return ImageDimensions(width, height)
        elif chunk == b"VP8L":
            if len(buf) < 25:
                return None
            bits = struct.unpack("<I", buf[21:25])[0]
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return ImageDimensions(width, height)
        elif chunk == b"VP8X":
            if len(buf) < 30:
                return None
            width = (buf[24] | (buf[25] << 8) | (buf[26] << 16)) + 1
            height = (buf[27] | (buf[28] << 8) | (buf[29] << 16)) + 1
            return ImageDimensions(width, height)
        return None
    except Exception:
        return None


def get_image_dimensions(base64_data: str, mime_type: str) -> ImageDimensions | None:
    if mime_type == "image/png":
        return get_png_dimensions(base64_data)
    if mime_type == "image/jpeg":
        return get_jpeg_dimensions(base64_data)
    if mime_type == "image/gif":
        return get_gif_dimensions(base64_data)
    if mime_type == "image/webp":
        return get_webp_dimensions(base64_data)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_image_rows(
    image_dims: ImageDimensions,
    target_width_cells: int,
    cell_dims: CellDimensions | None = None,
) -> int:
    if cell_dims is None:
        cell_dims = _cell_dimensions
    target_width_px = target_width_cells * cell_dims.width_px
    scale = target_width_px / image_dims.width_px
    scaled_height_px = image_dims.height_px * scale
    rows = int((scaled_height_px + cell_dims.height_px - 1) // cell_dims.height_px)
    return max(1, rows)


class _RenderResult:
    __slots__ = ("sequence", "rows", "image_id")

    def __init__(self, sequence: str, rows: int, image_id: int | None = None) -> None:
        self.sequence = sequence
        self.rows = rows
        self.image_id = image_id


def render_image(
    base64_data: str,
    image_dims: ImageDimensions,
    options: ImageRenderOptions | None = None,
) -> _RenderResult | None:
    """Render image using the best available protocol."""
    caps = get_capabilities()
    if not caps.images:
        return None

    opts = options or ImageRenderOptions()
    max_width = opts.max_width_cells or 80
    rows = calculate_image_rows(image_dims, max_width, get_cell_dimensions())

    if caps.images == "kitty":
        seq = encode_kitty(base64_data, columns=max_width, rows=rows, image_id=opts.image_id)
        return _RenderResult(seq, rows, opts.image_id)

    if caps.images == "iterm2":
        seq = encode_iterm2(
            base64_data,
            width=max_width,
            height="auto",
            preserve_aspect_ratio=opts.preserve_aspect_ratio,
        )
        return _RenderResult(seq, rows)

    return None


_KITTY_PREFIX = "\x1b_G"
_ITERM2_PREFIX = "\x1b]1337;File="


def is_image_line(line: str) -> bool:
    """Return True if the line contains an inline image sequence."""
    if line.startswith(_KITTY_PREFIX) or line.startswith(_ITERM2_PREFIX):
        return True
    return _KITTY_PREFIX in line or _ITERM2_PREFIX in line


def image_fallback(
    mime_type: str,
    dimensions: ImageDimensions | None = None,
    filename: str | None = None,
) -> str:
    parts: list[str] = []
    if filename:
        parts.append(filename)
    parts.append(f"[{mime_type}]")
    if dimensions:
        parts.append(f"{dimensions.width_px}x{dimensions.height_px}")
    return f"[Image: {' '.join(parts)}]"
