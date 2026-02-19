"""
EditorComponent Protocol — mirrors packages/tui/src/editor-component.ts

Defines the interface that custom editor components must implement to integrate
with the pi coding agent TUI.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .autocomplete import AutocompleteProvider


@runtime_checkable
class EditorComponent(Protocol):
    """
    Interface for custom editor components.

    Allows extensions to provide their own editor implementation
    (e.g., vim mode, emacs mode, custom keybindings) while maintaining
    compatibility with the core application.

    Mirrors EditorComponent in editor-component.ts.
    """

    # ─── Core text access (required) ────────────────────────────────────────

    def get_text(self) -> str:
        """Get the current text content."""
        ...

    def set_text(self, text: str) -> None:
        """Set the text content."""
        ...

    def handle_input(self, data: str) -> None:
        """Handle raw terminal input (key presses, paste sequences, etc.)."""
        ...

    # ─── Callbacks (required) ───────────────────────────────────────────────

    on_submit: Callable[[str], None] | None
    """Called when user submits (e.g., Enter key)."""

    on_change: Callable[[str], None] | None
    """Called when text changes."""

    # ─── Rendering (required — from Component protocol) ─────────────────────

    def render(self, width: int) -> list[str]:
        """Render the component to lines for the given viewport width."""
        ...

    def invalidate(self) -> None:
        """Invalidate any cached rendering state."""
        ...

    # ─── History support (optional) ─────────────────────────────────────────

    def add_to_history(self, text: str) -> None:
        """Add text to history for up/down navigation."""
        ...

    # ─── Advanced text manipulation (optional) ──────────────────────────────

    def insert_text_at_cursor(self, text: str) -> None:
        """Insert text at current cursor position."""
        ...

    def get_expanded_text(self) -> str:
        """Get text with any markers expanded. Falls back to get_text()."""
        ...

    # ─── Autocomplete support (optional) ────────────────────────────────────

    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        """Set the autocomplete provider."""
        ...

    # ─── Appearance (optional) ──────────────────────────────────────────────

    border_color: Callable[[str], str] | None
    """Border color function."""

    def set_padding_x(self, padding: int) -> None:
        """Set horizontal padding."""
        ...

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        """Set max visible items in autocomplete dropdown."""
        ...
