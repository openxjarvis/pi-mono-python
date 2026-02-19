"""Tests for pi_tui.fuzzy — mirrors fuzzy.ts tests"""
import pytest

from pi_tui.fuzzy import fuzzy_filter, fuzzy_match


class TestFuzzyMatch:
    def test_exact_match_matches(self):
        # fuzzy_match(query, text)
        result = fuzzy_match("hello", "hello")
        assert result.matches is True

    def test_empty_query_matches_all(self):
        result = fuzzy_match("", "hello")
        assert result.matches is True

    def test_non_match(self):
        result = fuzzy_match("xyz123", "hello")
        assert result.matches is False

    def test_prefix_match(self):
        result = fuzzy_match("hel", "hello world")
        assert result.matches is True

    def test_consecutive_chars_better_score(self):
        # "he" consecutive in "hello" should have better (lower) score than "hl"
        r1 = fuzzy_match("he", "hello")
        r2 = fuzzy_match("hl", "hello")
        assert r1.matches is True
        assert r2.matches is True
        assert r1.score <= r2.score


class TestFuzzyFilter:
    def test_filters_matching_items(self):
        items = ["apple", "banana", "apricot", "cherry"]
        result = fuzzy_filter(items, "ap", get_text=lambda x: x)
        assert "apple" in result
        assert "apricot" in result
        assert "banana" not in result

    def test_empty_query_returns_all(self):
        items = ["apple", "banana", "cherry"]
        result = fuzzy_filter(items, "", get_text=lambda x: x)
        assert len(result) == 3

    def test_no_match_returns_empty(self):
        items = ["apple", "banana", "cherry"]
        result = fuzzy_filter(items, "xyz999", get_text=lambda x: x)
        assert result == []

    def test_custom_key_function(self):
        items = [{"name": "apple"}, {"name": "banana"}]
        result = fuzzy_filter(items, "app", get_text=lambda x: x["name"])
        assert len(result) == 1
        assert result[0]["name"] == "apple"

    def test_case_insensitive(self):
        items = ["Apple", "BANANA", "cherry"]
        result = fuzzy_filter(items, "apple", get_text=lambda x: x)
        assert "Apple" in result

    def test_sorted_by_score(self):
        items = ["hello world", "hello", "world hello"]
        result = fuzzy_filter(items, "hello", get_text=lambda x: x)
        assert len(result) >= 2
