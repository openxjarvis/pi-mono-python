"""
Keyboard input handling — mirrors packages/tui/src/keys.ts

Supports both legacy terminal sequences and Kitty keyboard protocol.
See: https://sw.kovidgoyal.net/kitty/keyboard-protocol/

API:
- matches_key(data, key_id) — check if input matches a key identifier
- parse_key(data) — parse input and return the key identifier string
- is_key_release(data) — check if event is a key release (Kitty flag 2)
- is_key_repeat(data) — check if event is a key repeat (Kitty flag 2)
- set_kitty_protocol_active(active) — set global Kitty protocol state
- is_kitty_protocol_active() — query global Kitty protocol state
- KEY — helper constants for common keys
"""
from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Global Kitty Protocol state
# ─────────────────────────────────────────────────────────────────────────────

_kitty_protocol_active = False


def set_kitty_protocol_active(active: bool) -> None:
    global _kitty_protocol_active
    _kitty_protocol_active = active


def is_kitty_protocol_active() -> bool:
    return _kitty_protocol_active


# ─────────────────────────────────────────────────────────────────────────────
# KeyId type alias — in Python we just use str
# ─────────────────────────────────────────────────────────────────────────────

KeyId = str

# ─────────────────────────────────────────────────────────────────────────────
# Key helper — mirrors the Key object in keys.ts
# ─────────────────────────────────────────────────────────────────────────────

class _KeyHelper:
    """Helper object for creating typed key identifier strings with autocomplete."""

    # Special keys
    escape = "escape"
    esc = "esc"
    enter = "enter"
    ret = "return"
    tab = "tab"
    space = "space"
    backspace = "backspace"
    delete = "delete"
    insert = "insert"
    clear = "clear"
    home = "home"
    end = "end"
    page_up = "pageUp"
    page_down = "pageDown"
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"

    # Symbol keys
    backtick = "`"
    hyphen = "-"
    equals = "="
    leftbracket = "["
    rightbracket = "]"
    backslash = "\\"
    semicolon = ";"
    quote = "'"
    comma = ","
    period = "."
    slash = "/"

    @staticmethod
    def ctrl(key: str) -> str:
        return f"ctrl+{key}"

    @staticmethod
    def shift(key: str) -> str:
        return f"shift+{key}"

    @staticmethod
    def alt(key: str) -> str:
        return f"alt+{key}"

    @staticmethod
    def ctrl_shift(key: str) -> str:
        return f"ctrl+shift+{key}"

    @staticmethod
    def ctrl_alt(key: str) -> str:
        return f"ctrl+alt+{key}"

    @staticmethod
    def shift_alt(key: str) -> str:
        return f"shift+alt+{key}"

    @staticmethod
    def ctrl_shift_alt(key: str) -> str:
        return f"ctrl+shift+alt+{key}"


KEY = _KeyHelper()

# ─────────────────────────────────────────────────────────────────────────────
# Constants — mirrors the TS constants
# ─────────────────────────────────────────────────────────────────────────────

SYMBOL_KEYS: frozenset[str] = frozenset([
    "`", "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/",
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_",
    "+", "|", "~", "{", "}", ":", "<", ">", "?",
])

_MOD_SHIFT = 1
_MOD_ALT = 2
_MOD_CTRL = 4
_LOCK_MASK = 64 + 128

_CP_ESCAPE = 27
_CP_TAB = 9
_CP_ENTER = 13
_CP_SPACE = 32
_CP_BACKSPACE = 127
_CP_KP_ENTER = 57414

_CP_UP = -1
_CP_DOWN = -2
_CP_RIGHT = -3
_CP_LEFT = -4

_CP_DELETE = -10
_CP_INSERT = -11
_CP_PAGE_UP = -12
_CP_PAGE_DOWN = -13
_CP_HOME = -14
_CP_END = -15

_LEGACY_KEY_SEQS: dict[str, list[str]] = {
    "up":       ["\x1b[A", "\x1bOA"],
    "down":     ["\x1b[B", "\x1bOB"],
    "right":    ["\x1b[C", "\x1bOC"],
    "left":     ["\x1b[D", "\x1bOD"],
    "home":     ["\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"],
    "end":      ["\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"],
    "insert":   ["\x1b[2~"],
    "delete":   ["\x1b[3~"],
    "pageUp":   ["\x1b[5~", "\x1b[[5~"],
    "pageDown": ["\x1b[6~", "\x1b[[6~"],
    "clear":    ["\x1b[E", "\x1bOE"],
    "f1":  ["\x1bOP", "\x1b[11~", "\x1b[[A"],
    "f2":  ["\x1bOQ", "\x1b[12~", "\x1b[[B"],
    "f3":  ["\x1bOR", "\x1b[13~", "\x1b[[C"],
    "f4":  ["\x1bOS", "\x1b[14~", "\x1b[[D"],
    "f5":  ["\x1b[15~", "\x1b[[E"],
    "f6":  ["\x1b[17~"],
    "f7":  ["\x1b[18~"],
    "f8":  ["\x1b[19~"],
    "f9":  ["\x1b[20~"],
    "f10": ["\x1b[21~"],
    "f11": ["\x1b[23~"],
    "f12": ["\x1b[24~"],
}

_LEGACY_SHIFT_SEQS: dict[str, list[str]] = {
    "up":       ["\x1b[a"],
    "down":     ["\x1b[b"],
    "right":    ["\x1b[c"],
    "left":     ["\x1b[d"],
    "clear":    ["\x1b[e"],
    "insert":   ["\x1b[2$"],
    "delete":   ["\x1b[3$"],
    "pageUp":   ["\x1b[5$"],
    "pageDown": ["\x1b[6$"],
    "home":     ["\x1b[7$"],
    "end":      ["\x1b[8$"],
}

_LEGACY_CTRL_SEQS: dict[str, list[str]] = {
    "up":       ["\x1bOa"],
    "down":     ["\x1bOb"],
    "right":    ["\x1bOc"],
    "left":     ["\x1bOd"],
    "clear":    ["\x1bOe"],
    "insert":   ["\x1b[2^"],
    "delete":   ["\x1b[3^"],
    "pageUp":   ["\x1b[5^"],
    "pageDown": ["\x1b[6^"],
    "home":     ["\x1b[7^"],
    "end":      ["\x1b[8^"],
}

_LEGACY_SEQ_KEY_IDS: dict[str, str] = {
    "\x1bOA": "up",
    "\x1bOB": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1b[E": "clear",
    "\x1bOE": "clear",
    "\x1bOe": "ctrl+clear",
    "\x1b[e": "shift+clear",
    "\x1b[2~": "insert",
    "\x1b[2$": "shift+insert",
    "\x1b[2^": "ctrl+insert",
    "\x1b[3$": "shift+delete",
    "\x1b[3^": "ctrl+delete",
    "\x1b[[5~": "pageUp",
    "\x1b[[6~": "pageDown",
    "\x1b[a": "shift+up",
    "\x1b[b": "shift+down",
    "\x1b[c": "shift+right",
    "\x1b[d": "shift+left",
    "\x1bOa": "ctrl+up",
    "\x1bOb": "ctrl+down",
    "\x1bOc": "ctrl+right",
    "\x1bOd": "ctrl+left",
    "\x1b[5$": "shift+pageUp",
    "\x1b[6$": "shift+pageDown",
    "\x1b[7$": "shift+home",
    "\x1b[8$": "shift+end",
    "\x1b[5^": "ctrl+pageUp",
    "\x1b[6^": "ctrl+pageDown",
    "\x1b[7^": "ctrl+home",
    "\x1b[8^": "ctrl+end",
    "\x1bOP": "f1",
    "\x1bOQ": "f2",
    "\x1bOR": "f3",
    "\x1bOS": "f4",
    "\x1b[11~": "f1",
    "\x1b[12~": "f2",
    "\x1b[13~": "f3",
    "\x1b[14~": "f4",
    "\x1b[[A": "f1",
    "\x1b[[B": "f2",
    "\x1b[[C": "f3",
    "\x1b[[D": "f4",
    "\x1b[[E": "f5",
    "\x1b[15~": "f5",
    "\x1b[17~": "f6",
    "\x1b[18~": "f7",
    "\x1b[19~": "f8",
    "\x1b[20~": "f9",
    "\x1b[21~": "f10",
    "\x1b[23~": "f11",
    "\x1b[24~": "f12",
    "\x1bb": "alt+left",
    "\x1bf": "alt+right",
    "\x1bp": "alt+up",
    "\x1bn": "alt+down",
}

# ─────────────────────────────────────────────────────────────────────────────
# Key release / repeat detection
# ─────────────────────────────────────────────────────────────────────────────

_RELEASE_SUFFIXES = (":3u", ":3~", ":3A", ":3B", ":3C", ":3D", ":3H", ":3F")
_REPEAT_SUFFIXES = (":2u", ":2~", ":2A", ":2B", ":2C", ":2D", ":2H", ":2F")


def is_key_release(data: str) -> bool:
    """Check if data is a Kitty key-release event."""
    if "\x1b[200~" in data:
        return False
    return any(data.endswith(s) or s in data for s in _RELEASE_SUFFIXES)


def is_key_repeat(data: str) -> bool:
    """Check if data is a Kitty key-repeat event."""
    if "\x1b[200~" in data:
        return False
    return any(data.endswith(s) or s in data for s in _REPEAT_SUFFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Kitty sequence parsing
# ─────────────────────────────────────────────────────────────────────────────

KeyEventType = str  # "press" | "repeat" | "release"


class _ParsedKitty:
    __slots__ = ("codepoint", "shifted_key", "base_layout_key", "modifier", "event_type")

    def __init__(
        self,
        codepoint: int,
        modifier: int,
        event_type: KeyEventType,
        shifted_key: int | None = None,
        base_layout_key: int | None = None,
    ) -> None:
        self.codepoint = codepoint
        self.shifted_key = shifted_key
        self.base_layout_key = base_layout_key
        self.modifier = modifier
        self.event_type = event_type


def _parse_event_type(s: str | None) -> KeyEventType:
    if not s:
        return "press"
    v = int(s)
    if v == 2:
        return "repeat"
    if v == 3:
        return "release"
    return "press"


_CSI_U_RE = re.compile(r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$")
_ARROW_MOD_RE = re.compile(r"^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$")
_FUNC_MOD_RE = re.compile(r"^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$")
_HOME_END_MOD_RE = re.compile(r"^\x1b\[1;(\d+)(?::(\d+))?([HF])$")

_FUNC_CODEPOINTS: dict[int, int] = {
    2: _CP_INSERT, 3: _CP_DELETE,
    5: _CP_PAGE_UP, 6: _CP_PAGE_DOWN,
    7: _CP_HOME, 8: _CP_END,
}
_ARROW_CODEPOINTS_MAP: dict[str, int] = {"A": _CP_UP, "B": _CP_DOWN, "C": _CP_RIGHT, "D": _CP_LEFT}


def _parse_kitty(data: str) -> _ParsedKitty | None:
    m = _CSI_U_RE.match(data)
    if m:
        cp = int(m.group(1))
        shifted = int(m.group(2)) if m.group(2) and m.group(2) != "" else None
        base = int(m.group(3)) if m.group(3) else None
        mod_val = int(m.group(4)) if m.group(4) else 1
        et = _parse_event_type(m.group(5))
        return _ParsedKitty(cp, mod_val - 1, et, shifted, base)

    m = _ARROW_MOD_RE.match(data)
    if m:
        mod_val = int(m.group(1))
        et = _parse_event_type(m.group(2))
        cp = _ARROW_CODEPOINTS_MAP[m.group(3)]
        return _ParsedKitty(cp, mod_val - 1, et)

    m = _FUNC_MOD_RE.match(data)
    if m:
        key_num = int(m.group(1))
        mod_val = int(m.group(2)) if m.group(2) else 1
        et = _parse_event_type(m.group(3))
        cp = _FUNC_CODEPOINTS.get(key_num)
        if cp is not None:
            return _ParsedKitty(cp, mod_val - 1, et)

    m = _HOME_END_MOD_RE.match(data)
    if m:
        mod_val = int(m.group(1))
        et = _parse_event_type(m.group(2))
        cp = _CP_HOME if m.group(3) == "H" else _CP_END
        return _ParsedKitty(cp, mod_val - 1, et)

    return None


def _matches_kitty(data: str, expected_cp: int, expected_mod: int) -> bool:
    parsed = _parse_kitty(data)
    if not parsed:
        return False
    actual_mod = parsed.modifier & ~_LOCK_MASK
    if actual_mod != (expected_mod & ~_LOCK_MASK):
        return False
    if parsed.codepoint == expected_cp:
        return True
    if parsed.base_layout_key == expected_cp:
        cp = parsed.codepoint
        is_latin = 97 <= cp <= 122
        is_known_sym = chr(cp) in SYMBOL_KEYS if 0 <= cp <= 0xFFFF else False
        if not is_latin and not is_known_sym:
            return True
    return False


def _matches_modify_other_keys(data: str, expected_keycode: int, expected_mod: int) -> bool:
    m = re.match(r"^\x1b\[27;(\d+);(\d+)~$", data)
    if not m:
        return False
    mod_val = int(m.group(1)) - 1
    keycode = int(m.group(2))
    return keycode == expected_keycode and mod_val == expected_mod


# ─────────────────────────────────────────────────────────────────────────────
# Key ID parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_key_id(key_id: str) -> tuple[str, bool, bool, bool] | None:
    """Return (key, ctrl, shift, alt) or None."""
    parts = key_id.lower().split("+")
    key = parts[-1] if parts else ""
    if not key:
        return None
    ctrl = "ctrl" in parts
    shift = "shift" in parts
    alt = "alt" in parts
    return key, ctrl, shift, alt


def _raw_ctrl_char(key: str) -> str | None:
    """Get control character for key (ctrl+a → chr(1), etc.)."""
    ch = key.lower()
    code = ord(ch)
    if (97 <= code <= 122) or ch in ("[", "\\", "]", "_"):
        return chr(code & 0x1f)
    if ch == "-":
        return chr(31)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# matches_key — mirrors matchesKey() in keys.ts
# ─────────────────────────────────────────────────────────────────────────────

def matches_key(data: str, key_id: KeyId) -> bool:
    """
    Check if *data* (raw terminal input) matches the given key identifier.
    Mirrors matchesKey() in keys.ts.
    """
    parsed = _parse_key_id(key_id)
    if not parsed:
        return False
    key, ctrl, shift, alt = parsed

    modifier = 0
    if shift:
        modifier |= _MOD_SHIFT
    if alt:
        modifier |= _MOD_ALT
    if ctrl:
        modifier |= _MOD_CTRL

    if key in ("escape", "esc"):
        if modifier != 0:
            return False
        return data == "\x1b" or _matches_kitty(data, _CP_ESCAPE, 0)

    if key == "space":
        if not _kitty_protocol_active:
            if ctrl and not alt and not shift and data == "\x00":
                return True
            if alt and not ctrl and not shift and data == "\x1b ":
                return True
        if modifier == 0:
            return data == " " or _matches_kitty(data, _CP_SPACE, 0)
        return _matches_kitty(data, _CP_SPACE, modifier)

    if key == "tab":
        if shift and not ctrl and not alt:
            return data == "\x1b[Z" or _matches_kitty(data, _CP_TAB, _MOD_SHIFT)
        if modifier == 0:
            return data == "\t" or _matches_kitty(data, _CP_TAB, 0)
        return _matches_kitty(data, _CP_TAB, modifier)

    if key in ("enter", "return"):
        if shift and not ctrl and not alt:
            if (_matches_kitty(data, _CP_ENTER, _MOD_SHIFT) or
                    _matches_kitty(data, _CP_KP_ENTER, _MOD_SHIFT)):
                return True
            if _matches_modify_other_keys(data, _CP_ENTER, _MOD_SHIFT):
                return True
            if _kitty_protocol_active:
                return data in ("\x1b\r", "\n")
            return False
        if alt and not ctrl and not shift:
            if (_matches_kitty(data, _CP_ENTER, _MOD_ALT) or
                    _matches_kitty(data, _CP_KP_ENTER, _MOD_ALT)):
                return True
            if _matches_modify_other_keys(data, _CP_ENTER, _MOD_ALT):
                return True
            if not _kitty_protocol_active:
                return data == "\x1b\r"
            return False
        if modifier == 0:
            return (
                data == "\r" or
                (not _kitty_protocol_active and data == "\n") or
                data == "\x1bOM" or
                _matches_kitty(data, _CP_ENTER, 0) or
                _matches_kitty(data, _CP_KP_ENTER, 0)
            )
        return (
            _matches_kitty(data, _CP_ENTER, modifier) or
            _matches_kitty(data, _CP_KP_ENTER, modifier)
        )

    if key == "backspace":
        if alt and not ctrl and not shift:
            if data in ("\x1b\x7f", "\x1b\x08"):
                return True
            return _matches_kitty(data, _CP_BACKSPACE, _MOD_ALT)
        if modifier == 0:
            return data in ("\x7f", "\x08") or _matches_kitty(data, _CP_BACKSPACE, 0)
        return _matches_kitty(data, _CP_BACKSPACE, modifier)

    if key == "insert":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["insert"] or _matches_kitty(data, _CP_INSERT, 0)
        if _matches_legacy_modifier(data, "insert", modifier):
            return True
        return _matches_kitty(data, _CP_INSERT, modifier)

    if key == "delete":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["delete"] or _matches_kitty(data, _CP_DELETE, 0)
        if _matches_legacy_modifier(data, "delete", modifier):
            return True
        return _matches_kitty(data, _CP_DELETE, modifier)

    if key == "clear":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS.get("clear", [])
        return _matches_legacy_modifier(data, "clear", modifier)

    if key == "home":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["home"] or _matches_kitty(data, _CP_HOME, 0)
        if _matches_legacy_modifier(data, "home", modifier):
            return True
        return _matches_kitty(data, _CP_HOME, modifier)

    if key == "end":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["end"] or _matches_kitty(data, _CP_END, 0)
        if _matches_legacy_modifier(data, "end", modifier):
            return True
        return _matches_kitty(data, _CP_END, modifier)

    if key == "pageup":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["pageUp"] or _matches_kitty(data, _CP_PAGE_UP, 0)
        if _matches_legacy_modifier(data, "pageUp", modifier):
            return True
        return _matches_kitty(data, _CP_PAGE_UP, modifier)

    if key == "pagedown":
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["pageDown"] or _matches_kitty(data, _CP_PAGE_DOWN, 0)
        if _matches_legacy_modifier(data, "pageDown", modifier):
            return True
        return _matches_kitty(data, _CP_PAGE_DOWN, modifier)

    if key == "up":
        if alt and not ctrl and not shift:
            return data == "\x1bp" or _matches_kitty(data, _CP_UP, _MOD_ALT)
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["up"] or _matches_kitty(data, _CP_UP, 0)
        if _matches_legacy_modifier(data, "up", modifier):
            return True
        return _matches_kitty(data, _CP_UP, modifier)

    if key == "down":
        if alt and not ctrl and not shift:
            return data == "\x1bn" or _matches_kitty(data, _CP_DOWN, _MOD_ALT)
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["down"] or _matches_kitty(data, _CP_DOWN, 0)
        if _matches_legacy_modifier(data, "down", modifier):
            return True
        return _matches_kitty(data, _CP_DOWN, modifier)

    if key == "left":
        if alt and not ctrl and not shift:
            return (
                data == "\x1b[1;3D" or
                (not _kitty_protocol_active and data == "\x1bB") or
                data == "\x1bb" or
                _matches_kitty(data, _CP_LEFT, _MOD_ALT)
            )
        if ctrl and not alt and not shift:
            return (
                data == "\x1b[1;5D" or
                _matches_legacy_modifier(data, "left", _MOD_CTRL) or
                _matches_kitty(data, _CP_LEFT, _MOD_CTRL)
            )
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["left"] or _matches_kitty(data, _CP_LEFT, 0)
        if _matches_legacy_modifier(data, "left", modifier):
            return True
        return _matches_kitty(data, _CP_LEFT, modifier)

    if key == "right":
        if alt and not ctrl and not shift:
            return (
                data == "\x1b[1;3C" or
                (not _kitty_protocol_active and data == "\x1bF") or
                data == "\x1bf" or
                _matches_kitty(data, _CP_RIGHT, _MOD_ALT)
            )
        if ctrl and not alt and not shift:
            return (
                data == "\x1b[1;5C" or
                _matches_legacy_modifier(data, "right", _MOD_CTRL) or
                _matches_kitty(data, _CP_RIGHT, _MOD_CTRL)
            )
        if modifier == 0:
            return data in _LEGACY_KEY_SEQS["right"] or _matches_kitty(data, _CP_RIGHT, 0)
        if _matches_legacy_modifier(data, "right", modifier):
            return True
        return _matches_kitty(data, _CP_RIGHT, modifier)

    if key in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"):
        if modifier != 0:
            return False
        return data in _LEGACY_KEY_SEQS.get(key, [])

    # Single-letter and symbol keys
    if len(key) == 1 and (("a" <= key <= "z") or key in SYMBOL_KEYS):
        codepoint = ord(key)
        raw_ctrl = _raw_ctrl_char(key)

        if ctrl and alt and not shift and not _kitty_protocol_active and raw_ctrl:
            return data == f"\x1b{raw_ctrl}"

        if alt and not ctrl and not shift and not _kitty_protocol_active and "a" <= key <= "z":
            if data == f"\x1b{key}":
                return True

        if ctrl and not shift and not alt:
            if raw_ctrl and data == raw_ctrl:
                return True
            return _matches_kitty(data, codepoint, _MOD_CTRL)

        if ctrl and shift and not alt:
            return _matches_kitty(data, codepoint, _MOD_SHIFT + _MOD_CTRL)

        if shift and not ctrl and not alt:
            if data == key.upper():
                return True
            return _matches_kitty(data, codepoint, _MOD_SHIFT)

        if modifier != 0:
            return _matches_kitty(data, codepoint, modifier)

        return data == key or _matches_kitty(data, codepoint, 0)

    return False


def _matches_legacy_modifier(data: str, key_name: str, modifier: int) -> bool:
    if modifier == _MOD_SHIFT:
        return data in _LEGACY_SHIFT_SEQS.get(key_name, [])
    if modifier == _MOD_CTRL:
        return data in _LEGACY_CTRL_SEQS.get(key_name, [])
    return False


# ─────────────────────────────────────────────────────────────────────────────
# parse_key — mirrors parseKey() in keys.ts
# ─────────────────────────────────────────────────────────────────────────────

def parse_key(data: str) -> str | None:
    """
    Parse raw terminal input and return a key identifier string, or None.
    Mirrors parseKey() in keys.ts.
    """
    kitty = _parse_kitty(data)
    if kitty:
        cp = kitty.codepoint
        mod = kitty.modifier & ~_LOCK_MASK
        mods: list[str] = []
        if mod & _MOD_SHIFT:
            mods.append("shift")
        if mod & _MOD_CTRL:
            mods.append("ctrl")
        if mod & _MOD_ALT:
            mods.append("alt")

        is_latin = 97 <= cp <= 122
        is_known_sym = chr(cp) in SYMBOL_KEYS if 0 <= cp <= 0xFFFF else False
        effective_cp = cp if (is_latin or is_known_sym) else (kitty.base_layout_key or cp)

        key_name: str | None = None
        if effective_cp == _CP_ESCAPE:
            key_name = "escape"
        elif effective_cp == _CP_TAB:
            key_name = "tab"
        elif effective_cp in (_CP_ENTER, _CP_KP_ENTER):
            key_name = "enter"
        elif effective_cp == _CP_SPACE:
            key_name = "space"
        elif effective_cp == _CP_BACKSPACE:
            key_name = "backspace"
        elif effective_cp == _CP_DELETE:
            key_name = "delete"
        elif effective_cp == _CP_INSERT:
            key_name = "insert"
        elif effective_cp == _CP_HOME:
            key_name = "home"
        elif effective_cp == _CP_END:
            key_name = "end"
        elif effective_cp == _CP_PAGE_UP:
            key_name = "pageUp"
        elif effective_cp == _CP_PAGE_DOWN:
            key_name = "pageDown"
        elif effective_cp == _CP_UP:
            key_name = "up"
        elif effective_cp == _CP_DOWN:
            key_name = "down"
        elif effective_cp == _CP_LEFT:
            key_name = "left"
        elif effective_cp == _CP_RIGHT:
            key_name = "right"
        elif 97 <= effective_cp <= 122:
            key_name = chr(effective_cp)
        elif 0 <= effective_cp <= 0xFFFF and chr(effective_cp) in SYMBOL_KEYS:
            key_name = chr(effective_cp)

        if key_name:
            return "+".join(mods + [key_name]) if mods else key_name

    # Mode-aware legacy sequences
    if _kitty_protocol_active:
        if data in ("\x1b\r", "\n"):
            return "shift+enter"

    seq_id = _LEGACY_SEQ_KEY_IDS.get(data)
    if seq_id:
        return seq_id

    if data == "\x1b":
        return "escape"
    if data == "\x1c":
        return "ctrl+\\"
    if data == "\x1d":
        return "ctrl+]"
    if data == "\x1f":
        return "ctrl+-"
    if data == "\x1b\x1b":
        return "ctrl+alt+["
    if data == "\x1b\x1c":
        return "ctrl+alt+\\"
    if data == "\x1b\x1d":
        return "ctrl+alt+]"
    if data == "\x1b\x1f":
        return "ctrl+alt+-"
    if data == "\t":
        return "tab"
    if data == "\r" or (not _kitty_protocol_active and data == "\n") or data == "\x1bOM":
        return "enter"
    if data == "\x00":
        return "ctrl+space"
    if data == " ":
        return "space"
    if data in ("\x7f", "\x08"):
        return "backspace"
    if data == "\x1b[Z":
        return "shift+tab"
    if not _kitty_protocol_active and data == "\x1b\r":
        return "alt+enter"
    if not _kitty_protocol_active and data == "\x1b ":
        return "alt+space"
    if data in ("\x1b\x7f", "\x1b\x08"):
        return "alt+backspace"
    if not _kitty_protocol_active and data == "\x1bB":
        return "alt+left"
    if not _kitty_protocol_active and data == "\x1bF":
        return "alt+right"

    if not _kitty_protocol_active and len(data) == 2 and data[0] == "\x1b":
        code = ord(data[1])
        if 1 <= code <= 26:
            return f"ctrl+alt+{chr(code + 96)}"
        if 97 <= code <= 122:
            return f"alt+{chr(code)}"

    if data == "\x1b[A":
        return "up"
    if data == "\x1b[B":
        return "down"
    if data == "\x1b[C":
        return "right"
    if data == "\x1b[D":
        return "left"
    if data in ("\x1b[H", "\x1bOH"):
        return "home"
    if data in ("\x1b[F", "\x1bOF"):
        return "end"
    if data == "\x1b[3~":
        return "delete"
    if data == "\x1b[5~":
        return "pageUp"
    if data == "\x1b[6~":
        return "pageDown"

    if len(data) == 1:
        code = ord(data)
        if 1 <= code <= 26:
            return f"ctrl+{chr(code + 96)}"
        if 32 <= code <= 126:
            return data

    return None
