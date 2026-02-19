"""Tests for pi_tui.undo_stack — mirrors undo-stack.ts tests"""
import pytest

from pi_tui.undo_stack import UndoStack


class TestUndoStack:
    def test_initially_empty(self):
        stack: UndoStack[dict] = UndoStack()
        assert stack.pop() is None
        assert stack.length == 0

    def test_push_and_pop(self):
        stack: UndoStack[dict] = UndoStack()
        state = {"value": "hello", "cursor": 5}
        stack.push(state)
        result = stack.pop()
        assert result is not None
        assert result["value"] == "hello"
        assert result["cursor"] == 5

    def test_deep_copy_on_push(self):
        stack: UndoStack[dict] = UndoStack()
        state = {"value": "hello"}
        stack.push(state)
        state["value"] = "changed"  # mutate original
        result = stack.pop()
        assert result is not None
        assert result["value"] == "hello"  # snapshot is unchanged

    def test_multiple_pushes_lifo(self):
        stack: UndoStack[str] = UndoStack()
        stack.push("first")
        stack.push("second")
        stack.push("third")
        assert stack.pop() == "third"
        assert stack.pop() == "second"
        assert stack.pop() == "first"
        assert stack.pop() is None

    def test_clear(self):
        stack: UndoStack[str] = UndoStack()
        stack.push("a")
        stack.push("b")
        stack.clear()
        assert stack.length == 0
        assert stack.pop() is None

    def test_length(self):
        stack: UndoStack[int] = UndoStack()
        assert len(stack) == 0
        stack.push(1)
        stack.push(2)
        assert len(stack) == 2
