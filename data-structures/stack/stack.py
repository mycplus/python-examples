"""Two stack implementations and a bracket checker.

Run:  python3 stack.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class StackUnderflowError(IndexError):
    """Raised by pop() or peek() on an empty stack."""


class ArrayStack(Generic[T]):
    """A stack backed by a Python list, whose end is the top."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, value: T) -> None:
        self._items.append(value)

    def pop(self) -> T:
        if not self._items:
            raise StackUnderflowError("pop from an empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise StackUnderflowError("peek at an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class _Node(Generic[T]):
    value: T
    next: _Node[T] | None


class LinkedStack(Generic[T]):
    """A stack built on a singly linked list of nodes."""

    def __init__(self) -> None:
        self._top: _Node[T] | None = None
        self._size = 0

    def push(self, value: T) -> None:
        self._top = _Node(value, self._top)
        self._size += 1

    def pop(self) -> T:
        if self._top is None:
            raise StackUnderflowError("pop from an empty stack")
        node = self._top
        self._top = node.next
        self._size -= 1
        return node.value

    def peek(self) -> T:
        if self._top is None:
            raise StackUnderflowError("peek at an empty stack")
        return self._top.value

    def is_empty(self) -> bool:
        return self._top is None

    def __len__(self) -> int:
        return self._size


PAIRS = {")": "(", "]": "[", "}": "{"}


def balanced(text: str) -> bool:
    open_brackets: list[str] = []          # a plain list used as a stack
    for c in text:
        if c in "([{":
            open_brackets.append(c)
        elif c in PAIRS:
            if not open_brackets or open_brackets.pop() != PAIRS[c]:
                return False
    return not open_brackets


def main() -> None:
    s: ArrayStack[int] = ArrayStack()
    print("push 10 20 30")
    for v in (10, 20, 30):
        s.push(v)
    print(f"size={len(s)} top={s.peek()}")
    while not s.is_empty():
        print(f"pop {s.pop()}")
    print(f"empty={str(s.is_empty()).lower()}")
    try:
        s.pop()
        print("pop on empty: returned a value")
    except StackUnderflowError:
        print("pop on empty: underflow reported")

    for t in ("{[()()]}", "([)]", "((", "())", ""):
        print(f'balanced("{t}") = {str(balanced(t)).lower()}')


if __name__ == "__main__":
    main()
