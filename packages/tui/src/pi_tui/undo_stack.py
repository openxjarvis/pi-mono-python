"""
Generic undo stack — mirrors packages/tui/src/undo-stack.ts

Stores deep clones of state snapshots using copy.deepcopy.
"""
from __future__ import annotations

import copy
from typing import Generic, TypeVar

S = TypeVar("S")


class UndoStack(Generic[S]):
    """
    Generic undo stack with clone-on-push semantics.

    Stores deep copies of state snapshots.  Pop returns the snapshot directly
    (no re-copy) since it is already detached from the live state.
    Mirrors UndoStack<S> in undo-stack.ts.
    """

    def __init__(self) -> None:
        self._stack: list[S] = []

    def push(self, state: S) -> None:
        """Push a deep copy of *state* onto the stack."""
        self._stack.append(copy.deepcopy(state))

    def pop(self) -> S | None:
        """Pop and return the most recent snapshot, or None if empty."""
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        """Remove all snapshots."""
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)

    def __len__(self) -> int:
        return len(self._stack)
