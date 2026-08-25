"""
Clipboard utilities — mirrors packages/coding-agent/src/utils/clipboard.ts
and clipboard-image.ts.

Text: copies/pastes across platforms (macOS, Windows, Linux Wayland/X11/Termux).
Image: reads/writes clipboard images via platform-native tools.
Always emits OSC 52 as a text fallback (works over SSH/mosh).
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
_DEFAULT_LIST_TIMEOUT_S = 1
_DEFAULT_READ_TIMEOUT_S = 3
_DEFAULT_MAX_BUFFER = 50 * 1024 * 1024


@dataclass
class ClipboardImage:
    """Mirrors TS ClipboardImage type."""
    data: bytes
    mime_type: str


def _is_wayland_session() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY")) or os.environ.get("XDG_SESSION_TYPE") == "wayland"


def _base_mime(mime: str) -> str:
    return mime.split(";")[0].strip().lower()


def extension_for_image_mime_type(mime_type: str) -> str | None:
    """Return file extension for an image MIME type, or None."""
    mapping = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
    return mapping.get(_base_mime(mime_type))


def _select_preferred_image_mime(types: list[str]) -> str | None:
    normalized = [(t.strip(), _base_mime(t)) for t in types if t.strip()]
    for preferred in SUPPORTED_IMAGE_MIME_TYPES:
        for raw, base in normalized:
            if base == preferred:
                return raw
    for raw, base in normalized:
        if base.startswith("image/"):
            return raw
    return None


def _run_cmd(cmd: list[str], *, timeout: float = _DEFAULT_READ_TIMEOUT_S) -> tuple[bool, bytes]:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if r.returncode != 0:
            return False, b""
        return True, r.stdout
    except Exception:
        return False, b""


# ──────────────────── Text clipboard ──────────────────────────────────────────

def read_clipboard_text() -> str:
    """Read plain text from the system clipboard."""
    try:
        if sys.platform == "darwin":
            ok, data = _run_cmd(["pbpaste"])
        elif sys.platform == "win32":
            ok, data = _run_cmd(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])
        elif _is_wayland_session():
            ok, data = _run_cmd(["wl-paste", "--no-newline"])
            if not ok:
                ok, data = _run_cmd(["xclip", "-selection", "clipboard", "-o"])
        elif os.environ.get("TERMUX_VERSION"):
            ok, data = _run_cmd(["termux-clipboard-get"])
        else:
            ok, data = _run_cmd(["xclip", "-selection", "clipboard", "-o"])
            if not ok:
                ok, data = _run_cmd(["xsel", "-b", "-o"])
        if ok and data:
            return data.decode("utf-8", errors="replace")
    except Exception:
        return ""
    return ""


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard (OSC 52 + native tools)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sys.stdout.write(f"\x1b]52;c;{encoded}\x07")
    sys.stdout.flush()
    _try_native_clipboard(text)


def _try_native_clipboard(text: str) -> None:
    input_bytes = text.encode("utf-8")
    timeout = 5

    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=input_bytes, timeout=timeout, check=True)
        elif sys.platform == "win32":
            subprocess.run(["clip"], input=input_bytes, timeout=timeout, check=True)
        else:
            if os.environ.get("TERMUX_VERSION"):
                try:
                    subprocess.run(["termux-clipboard-set"], input=input_bytes, timeout=timeout, check=True)
                    return
                except Exception:
                    pass

            if _is_wayland_session():
                try:
                    subprocess.run(["which", "wl-copy"], capture_output=True, check=True)
                    proc = subprocess.Popen(
                        ["wl-copy"], stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    if proc.stdin:
                        proc.stdin.write(input_bytes)
                        proc.stdin.close()
                    return
                except Exception:
                    pass
                try:
                    subprocess.run(["xclip", "-selection", "clipboard"], input=input_bytes, timeout=timeout, check=True)
                    return
                except Exception:
                    pass
                subprocess.run(["xsel", "--clipboard", "--input"], input=input_bytes, timeout=timeout, check=True)
            else:
                try:
                    subprocess.run(["xclip", "-selection", "clipboard"], input=input_bytes, timeout=timeout, check=True)
                except Exception:
                    subprocess.run(["xsel", "--clipboard", "--input"], input=input_bytes, timeout=timeout, check=True)
    except Exception:
        pass


# ──────────────────── Image clipboard — read ──────────────────────────────────

def _read_image_macos() -> ClipboardImage | None:
    """Read image from macOS clipboard using osascript + pngpaste fallback."""
    ok, stdout = _run_cmd(
        ["osascript", "-e", 'clipboard info for (clipboard info for (the clipboard))'],
        timeout=_DEFAULT_LIST_TIMEOUT_S,
    )
    has_image = False
    if ok:
        text = stdout.decode("utf-8", errors="replace").lower()
        has_image = "«class png " in text or "«class tiff" in text or "picture" in text
    if not has_image:
        ok2, stdout2 = _run_cmd(
            ["osascript", "-e", 'the clipboard as «class PNGf»'],
            timeout=_DEFAULT_READ_TIMEOUT_S,
        )
        if ok2 and len(stdout2) > 100:
            return ClipboardImage(data=stdout2, mime_type="image/png")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ok3, _ = _run_cmd(["pngpaste", tmp_path], timeout=_DEFAULT_READ_TIMEOUT_S)
        if ok3:
            p = Path(tmp_path)
            if p.exists() and p.stat().st_size > 0:
                return ClipboardImage(data=p.read_bytes(), mime_type="image/png")
    except Exception:
        pass
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return None


def _read_image_wayland() -> ClipboardImage | None:
    """Read image from Wayland clipboard via wl-paste, fallback to xclip."""
    ok, stdout = _run_cmd(["wl-paste", "--list-types"], timeout=_DEFAULT_LIST_TIMEOUT_S)
    if ok:
        types = [t.strip() for t in stdout.decode("utf-8", errors="replace").splitlines() if t.strip()]
        selected = _select_preferred_image_mime(types)
        if selected:
            ok2, data = _run_cmd(["wl-paste", "--type", selected, "--no-newline"])
            if ok2 and data:
                return ClipboardImage(data=data, mime_type=_base_mime(selected))
    return _read_image_xclip()


def _read_image_xclip() -> ClipboardImage | None:
    """Read image from X11 clipboard via xclip."""
    ok, stdout = _run_cmd(
        ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
        timeout=_DEFAULT_LIST_TIMEOUT_S,
    )
    candidate_types: list[str] = []
    if ok:
        candidate_types = [t.strip() for t in stdout.decode("utf-8", errors="replace").splitlines() if t.strip()]

    preferred = _select_preferred_image_mime(candidate_types) if candidate_types else None
    try_types = ([preferred] + list(SUPPORTED_IMAGE_MIME_TYPES)) if preferred else list(SUPPORTED_IMAGE_MIME_TYPES)

    for mime in try_types:
        if not mime:
            continue
        ok2, data = _run_cmd(["xclip", "-selection", "clipboard", "-t", mime, "-o"])
        if ok2 and data:
            return ClipboardImage(data=data, mime_type=_base_mime(mime))
    return None


def _read_image_windows() -> ClipboardImage | None:
    """Read image from Windows clipboard via PowerShell."""
    ps_script = (
        "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
        "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
        "if ($img) { $ms = New-Object System.IO.MemoryStream; "
        "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); "
        "$ms.Position = 0; [System.Console]::OpenStandardOutput().Write($ms.ToArray(), 0, $ms.Length) }"
    )
    ok, data = _run_cmd(["powershell", "-NoProfile", "-Command", ps_script])
    if ok and data:
        return ClipboardImage(data=data, mime_type="image/png")
    return None


def read_clipboard_image() -> ClipboardImage | None:
    """Read an image from the system clipboard. Returns None if no image is available.

    Mirrors TS readClipboardImage().
    """
    if os.environ.get("TERMUX_VERSION"):
        return None

    try:
        if sys.platform == "darwin":
            return _read_image_macos()
        elif sys.platform == "win32":
            return _read_image_windows()
        elif _is_wayland_session():
            return _read_image_wayland()
        else:
            return _read_image_xclip()
    except Exception:
        return None


# ──────────────────── Image clipboard — write ─────────────────────────────────

def copy_image_to_clipboard(data: bytes, mime_type: str = "image/png") -> bool:
    """Write image bytes to the system clipboard. Returns True on success."""
    timeout = 5
    try:
        if sys.platform == "darwin":
            with tempfile.NamedTemporaryFile(suffix=f".{extension_for_image_mime_type(mime_type) or 'png'}", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                ext = extension_for_image_mime_type(mime_type) or "png"
                script = (
                    f'set the clipboard to (read POSIX file "{tmp_path}" as «class PNGf»)'
                    if ext == "png" else
                    f'set the clipboard to (read POSIX file "{tmp_path}" as TIFF picture)'
                )
                subprocess.run(["osascript", "-e", script], timeout=timeout, check=True)
                return True
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        elif sys.platform == "win32":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                ps = (
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
                    f"$img = [System.Drawing.Image]::FromFile('{tmp_path}'); "
                    "[System.Windows.Forms.Clipboard]::SetImage($img)"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=timeout, check=True)
                return True
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        else:
            if _is_wayland_session():
                proc = subprocess.Popen(
                    ["wl-copy", "--type", mime_type],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if proc.stdin:
                    proc.stdin.write(data)
                    proc.stdin.close()
                return proc.wait(timeout=timeout) == 0

            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", mime_type, "-i"],
                input=data,
                timeout=timeout,
                check=True,
            )
            return True

    except Exception:
        return False
