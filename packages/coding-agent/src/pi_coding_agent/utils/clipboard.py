"""
Clipboard utilities — mirrors packages/coding-agent/src/utils/clipboard.ts

Copies text to clipboard across platforms: macOS, Windows, Linux (Wayland/X11/Termux).
Always emits OSC 52 as a fallback (works over SSH/mosh).
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys


def _is_wayland_session() -> bool:
    """Check if running in a Wayland session."""
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def copy_to_clipboard(text: str) -> None:
    """
    Copy text to clipboard.
    Always emits OSC 52 escape sequence (works over SSH/mosh).
    Also tries native tools as best-effort for local sessions.

    Mirrors copyToClipboard() in TypeScript.
    """
    # Always emit OSC 52 — works over SSH/mosh, harmless locally
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sys.stdout.write(f"\x1b]52;c;{encoded}\x07")
    sys.stdout.flush()

    # Also try native tools (best effort for local sessions)
    _try_native_clipboard(text)


def _try_native_clipboard(text: str) -> None:
    """Attempt to write to native clipboard tools."""
    input_bytes = text.encode("utf-8")
    timeout = 5

    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=input_bytes, timeout=timeout, check=True)
        elif sys.platform == "win32":
            subprocess.run(["clip"], input=input_bytes, timeout=timeout, check=True)
        else:
            # Linux: try Termux, then Wayland or X11 tools
            if os.environ.get("TERMUX_VERSION"):
                try:
                    subprocess.run(
                        ["termux-clipboard-set"],
                        input=input_bytes,
                        timeout=timeout,
                        check=True,
                    )
                    return
                except Exception:
                    pass

            if _is_wayland_session():
                try:
                    # Verify wl-copy exists first
                    subprocess.run(
                        ["which", "wl-copy"],
                        capture_output=True,
                        check=True,
                    )
                    # wl-copy hangs with subprocess.run — use Popen + non-blocking write
                    proc = subprocess.Popen(
                        ["wl-copy"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if proc.stdin:
                        proc.stdin.write(input_bytes)
                        proc.stdin.close()
                    return
                except Exception:
                    pass
                # Wayland fallback: xclip / xsel (XWayland)
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=input_bytes,
                        timeout=timeout,
                        check=True,
                    )
                    return
                except Exception:
                    pass
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=input_bytes,
                    timeout=timeout,
                    check=True,
                )
            else:
                # X11 session
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=input_bytes,
                        timeout=timeout,
                        check=True,
                    )
                except Exception:
                    subprocess.run(
                        ["xsel", "--clipboard", "--input"],
                        input=input_bytes,
                        timeout=timeout,
                        check=True,
                    )
    except Exception:
        # Ignore failures — OSC 52 already emitted as fallback
        pass
