"""
Native modifier-key probing — Python port of native-modifiers.ts.

The TypeScript implementation loads platform `.node` addons. This port does
not ship or load Node binaries. On Windows it uses ctypes + Win32
`GetAsyncKeyState`. On every other platform it is a documented no-op that
returns False.
"""
from __future__ import annotations

import sys
from typing import Literal

ModifierKey = Literal["shift", "command", "control", "option"]

_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12  # Alt / Option
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C
_KEY_DOWN_MASK = 0x8000

_win32_get_async_key_state = None
if sys.platform == "win32":
    try:
        import ctypes

        _win32_get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        _win32_get_async_key_state.argtypes = [ctypes.c_int]
        _win32_get_async_key_state.restype = ctypes.c_short
    except Exception:
        _win32_get_async_key_state = None


def _is_vk_pressed(vk: int) -> bool:
    if _win32_get_async_key_state is None:
        return False
    try:
        return (_win32_get_async_key_state(vk) & _KEY_DOWN_MASK) != 0
    except Exception:
        return False


def is_native_modifier_pressed(key: ModifierKey) -> bool:
    """
    Return True when the named modifier is currently held.

    Windows: ctypes/Win32 GetAsyncKeyState.
    Other platforms: no-op fallback (always False). There is no equivalent
    of the TS darwin/win32 `.node` helpers in this Python port.
    """
    if sys.platform != "win32" or _win32_get_async_key_state is None:
        return False
    if key == "shift":
        return _is_vk_pressed(_VK_SHIFT)
    if key == "control":
        return _is_vk_pressed(_VK_CONTROL)
    if key == "option":
        return _is_vk_pressed(_VK_MENU)
    if key == "command":
        return _is_vk_pressed(_VK_LWIN) or _is_vk_pressed(_VK_RWIN)
    return False
