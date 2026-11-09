"""A recursive height on chains of growing depth: the shape sorted
insertion gives a binary search tree. CPython stops a recursion at
sys.getrecursionlimit() frames, 1,000 by default.

Run:  python3 recursive_height.py
"""
from __future__ import annotations

import sys


class Node:
    __slots__ = ("left", "right")

    def __init__(self, right: Node | None) -> None:
        self.left: Node | None = None
        self.right = right


def height(n: Node | None) -> int:
    """The same function as the C version: both subtrees, recursively."""
    return -1 if n is None else 1 + max(height(n.left), height(n.right))


def main() -> None:
    print(f"recursion limit: {sys.getrecursionlimit()}")
    for size in (500, 990, 1000, 10_000):
        root: Node | None = None
        for _ in range(size):           # linked directly: no O(n^2) insertion
            root = Node(root)
        try:
            print(f"n = {size:,}: height {height(root):,}")
        except RecursionError:
            print(f"n = {size:,}: RecursionError")


if __name__ == "__main__":
    main()
