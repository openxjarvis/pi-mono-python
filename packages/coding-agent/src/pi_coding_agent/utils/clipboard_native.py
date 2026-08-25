"""
Python equivalent of utils/clipboard-native.ts.

TypeScript loads the optional ``@mariozechner/clipboard`` native addon.
Python uses platform subprocess tools (pbcopy/pbpaste, wl-clipboard, xclip,
PowerShell) instead of compiling a ``.node`` binary.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Protocol


class ClipboardModule(Protocol):
    async def get_text(self) -> str: ...
    async def set_text(self, text: str) -> None: ...
    def has_image(self) -> bool: ...
    async def get_image_binary(self) -> list[int]: ...


def _has_display() -> bool:
    if sys.platform == "darwin" or sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _run(cmd: list[str], *, input_bytes: bytes | None = None, timeout: float = 3.0) -> bytes | None:
    try:
        result = subprocess.run(
            cmd,
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


class _SubprocessClipboard:
    """ClipboardModule implemented with OS clipboard CLIs."""

    async def get_text(self) -> str:
        if sys.platform == "darwin":
            data = _run(["pbpaste"])
        elif sys.platform == "win32":
            data = _run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])
        elif os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            data = _run(["wl-paste", "--no-newline"]) or _run(["xclip", "-selection", "clipboard", "-o"])
        else:
            data = _run(["xclip", "-selection", "clipboard", "-o"]) or _run(["xsel", "-b", "-o"])
        return (data or b"").decode("utf-8", errors="replace")

    async def set_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        if sys.platform == "darwin":
            _run(["pbcopy"], input_bytes=payload)
        elif sys.platform == "win32":
            _run(["clip"], input_bytes=payload)
        elif os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            if _run(["wl-copy"], input_bytes=payload) is None:
                _run(["xclip", "-selection", "clipboard"], input_bytes=payload)
        else:
            if _run(["xclip", "-selection", "clipboard"], input_bytes=payload) is None:
                _run(["xsel", "-b", "-i"], input_bytes=payload)

    def has_image(self) -> bool:
        if sys.platform == "darwin":
            data = _run(["osascript", "-e", "clipboard info"])
            return bool(data and b"picture" in data.lower())
        if os.environ.get("WAYLAND_DISPLAY"):
            listed = _run(["wl-paste", "--list-types"]) or b""
            return b"image/" in listed
        return False

    async def get_image_binary(self) -> list[int]:
        if sys.platform == "darwin":
            data = _run(["osascript", "-e", "the clipboard as «class PNGf»"])
            return list(data or b"")
        if os.environ.get("WAYLAND_DISPLAY"):
            data = _run(["wl-paste", "--type", "image/png"])
            return list(data or b"")
        data = _run(["xclip", "-selection", "clipboard", "-t", "image/png", "-o"])
        return list(data or b"")


def load_clipboard_native() -> ClipboardModule | None:
    """Load a clipboard implementation, or None when there is no display / Termux."""
    if os.environ.get("TERMUX_VERSION"):
        return None
    if not _has_display():
        return None
    return _SubprocessClipboard()


clipboard = load_clipboard_native()
