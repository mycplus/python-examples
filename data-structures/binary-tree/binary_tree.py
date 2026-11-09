"""A binary search tree with unique keys, and the article's demo.

Insert, search, remove and every traversal use loops, an explicit stack or
a queue: CPython's default recursion limit is 1,000 frames, and sorted
insertion builds a tree as deep as it is large.

Run:  python3 binary_tree.py
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, Generic, Protocol, TypeVar


class _Ordered(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


K = TypeVar("K", bound=_Ordered)


class _Node(Generic[K]):
    __slots__ = ("key", "left", "right")

    def __init__(self, key: K) -> None:
        self.key = key
        self.left: _Node[K] | None = None    # keys smaller than key
        self.right: _Node[K] | None = None   # keys larger than key


class BinarySearchTree(Generic[K]):
    """Keys are compared with <; two keys are equal when neither is smaller."""

    def __init__(self) -> None:
        self._root: _Node[K] | None = None
        self._size = 0

    def insert(self, key: K) -> bool:
        """Return False if the key was already present."""
        if self._root is None:
            self._root = _Node(key)
            self._size += 1
            return True
        n = self._root
        while True:
            if key < n.key:
                if n.left is None:
                    n.left = _Node(key)
                    break
                n = n.left
            elif n.key < key:
                if n.right is None:
                    n.right = _Node(key)
                    break
                n = n.right
            else:
                return False
        self._size += 1
        return True

    def __contains__(self, key: K) -> bool:
        n = self._root
        while n is not None:
            if key < n.key:
                n = n.left
            elif n.key < key:
                n = n.right
            else:
                return True
        return False

    def remove(self, key: K) -> bool:
        """Return False if the key was not present."""
        parent: _Node[K] | None = None
        n = self._root
        while n is not None and (key < n.key or n.key < key):
            parent = n
            n = n.left if key < n.key else n.right
        if n is None:
            return False
        if n.left is not None and n.right is not None:
            # Two children: copy the successor's key in, then remove the
            # successor, which has no left child.
            succ_parent, succ = n, n.right
            while succ.left is not None:
                succ_parent, succ = succ, succ.left
            n.key = succ.key
            parent, n = succ_parent, succ
        child = n.left if n.left is not None else n.right
        if parent is None:
            self._root = child
        elif parent.left is n:
            parent.left = child
        else:
            parent.right = child
        self._size -= 1
        return True

    def preorder(self) -> Iterator[K]:
        stack = [self._root] if self._root is not None else []
        while stack:
            n = stack.pop()
            yield n.key
            if n.right is not None:     # pushed first, so visited second
                stack.append(n.right)
            if n.left is not None:
                stack.append(n.left)

    def inorder(self) -> Iterator[K]:
        stack: list[_Node[K]] = []
        n = self._root
        while n is not None or stack:
            if n is not None:
                stack.append(n)
                n = n.left
            else:
                n = stack.pop()
                yield n.key
                n = n.right

    def postorder(self) -> Iterator[K]:
        """Preorder with the children swapped, reversed."""
        stack = [self._root] if self._root is not None else []
        out: list[K] = []
        while stack:
            n = stack.pop()
            out.append(n.key)
            if n.left is not None:
                stack.append(n.left)
            if n.right is not None:
                stack.append(n.right)
        return reversed(out)

    def level_order(self) -> Iterator[K]:
        queue = deque([self._root] if self._root is not None else [])
        while queue:
            n = queue.popleft()
            yield n.key
            if n.left is not None:
                queue.append(n.left)
            if n.right is not None:
                queue.append(n.right)

    def height(self) -> int:
        """Height in edges: -1 for an empty tree. Counts levels."""
        level = [self._root] if self._root is not None else []
        h = -1
        while level:
            h += 1
            level = [c for n in level for c in (n.left, n.right) if c is not None]
        return h

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[K]:
        return self.inorder()


def _line(label: str, walk: Callable[[], Iterator[int]]) -> str:
    return f"{label:<13}" + " ".join(str(k) for k in walk())


def main() -> None:
    tree: BinarySearchTree[int] = BinarySearchTree()
    for key in (50, 30, 70, 20, 40, 60, 80, 35, 45, 65):
        tree.insert(key)
    print("inserted:    50 30 70 20 40 60 80 35 45 65")
    print(_line("preorder:", tree.preorder))
    print(_line("inorder:", tree.inorder))
    print(_line("postorder:", tree.postorder))
    print(_line("level order:", tree.level_order))
    print(f"size {len(tree)}, height {tree.height()}")
    print(f"contains 45: {'yes' if 45 in tree else 'no'}, "
          f"contains 55: {'yes' if 55 in tree else 'no'}")
    print(f"insert 40 again: {int(tree.insert(40))}")

    tree.remove(30)
    print("\nremove 30 (two children)")
    print(_line("inorder:", tree.inorder))
    print(_line("level order:", tree.level_order))

    tree.remove(50)
    tree.remove(20)
    print("\nremove 50 (root), 20 (leaf)")
    print(_line("inorder:", tree.inorder))
    print(_line("level order:", tree.level_order))
    print(f"size {len(tree)}, height {tree.height()}")
    print(f"remove 99: {'removed' if tree.remove(99) else 'not found'}")


if __name__ == "__main__":
    main()
