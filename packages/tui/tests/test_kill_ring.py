"""Tests for pi_tui.kill_ring — mirrors kill-ring.ts tests"""
import pytest

from pi_tui.kill_ring import KillRing


class TestKillRing:
    def test_initially_empty(self):
        kr = KillRing()
        assert kr.peek() is None
        assert kr.length == 0

    def test_push_and_peek(self):
        kr = KillRing()
        kr.push("hello", prepend=False)
        assert kr.peek() == "hello"

    def test_multiple_pushes(self):
        kr = KillRing()
        kr.push("first", prepend=False)
        kr.push("second", prepend=False)
        assert kr.peek() == "second"
        assert kr.length == 2

    def test_rotate(self):
        kr = KillRing()
        kr.push("first", prepend=False)
        kr.push("second", prepend=False)
        kr.rotate()
        # After rotate, last is moved to front
        assert kr.peek() == "first"

    def test_accumulate_append(self):
        kr = KillRing()
        kr.push("hello", prepend=False)
        kr.push(" world", prepend=False, accumulate=True)
        assert kr.peek() == "hello world"
        assert kr.length == 1

    def test_accumulate_prepend(self):
        kr = KillRing()
        kr.push("world", prepend=False)
        kr.push("hello ", prepend=True, accumulate=True)
        assert kr.peek() == "hello world"
        assert kr.length == 1

    def test_no_accumulate_creates_new(self):
        kr = KillRing()
        kr.push("hello", prepend=False)
        kr.push("world", prepend=False, accumulate=False)
        assert kr.length == 2

    def test_rotate_single_item_noop(self):
        kr = KillRing()
        kr.push("only", prepend=False)
        kr.rotate()
        assert kr.peek() == "only"

    def test_empty_push_ignored(self):
        kr = KillRing()
        kr.push("", prepend=False)
        assert kr.length == 0
