"""Tests for pi_tui.autocomplete"""
import pytest

from pi_tui.autocomplete import (
    AutocompleteItem,
    CombinedAutocompleteProvider,
    SlashCommand,
)


class TestCombinedAutocompleteProvider:
    def _make_provider(self, commands=None):
        if commands is None:
            commands = [
                SlashCommand(name="help", description="Show help"),
                SlashCommand(name="clear", description="Clear screen"),
                SlashCommand(name="compact", description="Compact mode"),
            ]
        return CombinedAutocompleteProvider(
            commands=commands,
            base_path="/tmp",
        )

    def test_slash_command_suggestions(self):
        provider = self._make_provider()
        result = provider.get_suggestions(["/h"], 0, 2)
        assert result is not None
        assert len(result.items) > 0
        assert any(item.value == "help" or "/help" in item.value for item in result.items)

    def test_no_suggestions_for_regular_text(self):
        provider = self._make_provider()
        result = provider.get_suggestions(["hello world"], 0, 11)
        # No suggestions for non-slash, non-@ text
        assert result is None or len(result.items) == 0

    def test_slash_without_chars_shows_all(self):
        provider = self._make_provider()
        result = provider.get_suggestions(["/"], 0, 1)
        assert result is not None
        assert len(result.items) >= 3  # all commands

    def test_at_sign_suggestions(self):
        provider = self._make_provider()
        result = provider.get_suggestions(["@"], 0, 1)
        # May or may not have results depending on filesystem, but shouldn't crash
        # (result could be None or have items)

    def test_apply_completion_slash_command(self):
        provider = self._make_provider()
        item = AutocompleteItem(value="/help", label="help")
        result = provider.apply_completion(["/h"], 0, 2, item, "/h")
        assert result is not None
        assert result.lines is not None

    def test_prefix_extracted_correctly(self):
        provider = self._make_provider()
        result = provider.get_suggestions(["/cl"], 0, 3)
        if result:
            assert result.prefix.startswith("/")
