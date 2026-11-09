"""Run from the example directory:  python3 -m unittest discover -s tests"""
from __future__ import annotations

import io
import pathlib
import random
import unittest
from contextlib import redirect_stdout

# Run from the example directory, which puts binary_tree.py on sys.path.
import binary_tree
from binary_tree import BinarySearchTree

HERE = pathlib.Path(__file__).resolve().parent
KEYS = (50, 30, 70, 20, 40, 60, 80, 35, 45, 65)


def demo_tree() -> BinarySearchTree[int]:
    t: BinarySearchTree[int] = BinarySearchTree()
    for k in KEYS:
        t.insert(k)
    return t


class DemoTree(unittest.TestCase):
    def test_traversals(self) -> None:
        t = demo_tree()
        self.assertFalse(t.insert(40))
        self.assertEqual(list(t.preorder()), [50, 30, 20, 40, 35, 45, 70, 60, 65, 80])
        self.assertEqual(list(t.inorder()), [20, 30, 35, 40, 45, 50, 60, 65, 70, 80])
        self.assertEqual(list(t.postorder()), [20, 35, 45, 40, 30, 65, 60, 80, 70, 50])
        self.assertEqual(list(t.level_order()), list(KEYS))
        self.assertEqual((len(t), t.height()), (10, 3))

    def test_every_removal_case(self) -> None:
        t = demo_tree()
        for k in (65, 40, 45, 70, 80, 30, 50):  # leaf, two, left-only, two, left-only, right-only, root
            self.assertTrue(t.remove(k))
        self.assertFalse(t.remove(50))
        self.assertEqual(list(t), [20, 35, 60])
        self.assertEqual(len(t), 3)

    def test_empty_tree(self) -> None:
        t: BinarySearchTree[int] = BinarySearchTree()
        self.assertEqual((len(t), t.height()), (0, -1))
        self.assertNotIn(1, t)
        self.assertFalse(t.remove(1))
        for walk in (t.preorder, t.inorder, t.postorder, t.level_order):
            self.assertEqual(list(walk()), [])

    def test_strings(self) -> None:
        t: BinarySearchTree[str] = BinarySearchTree()
        for w in ("pear", "apple", "fig", "kiwi", "apple"):
            t.insert(w)
        self.assertEqual(list(t), ["apple", "fig", "kiwi", "pear"])

    def test_demo_output(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            binary_tree.main()
        expected = (HERE / "expected" / "binary_tree.txt").read_text()
        self.assertEqual(buf.getvalue(), expected)


class AgainstSet(unittest.TestCase):
    def test_random_operations(self) -> None:
        rng = random.Random(20261109)
        t: BinarySearchTree[int] = BinarySearchTree()
        model: set[int] = set()
        for op in range(200_000):
            key = rng.randrange(2000)
            action = rng.randrange(3)
            if action == 0:
                self.assertEqual(t.insert(key), key not in model)
                model.add(key)
            elif action == 1:
                self.assertEqual(t.remove(key), key in model)
                model.discard(key)
            else:
                self.assertEqual(key in t, key in model)
            if op % 1000 == 0:
                self.assertEqual(len(t), len(model))
                self.assertEqual(list(t), sorted(model))


class DeepChain(unittest.TestCase):
    def test_sorted_insertion_does_not_recurse(self) -> None:
        t: BinarySearchTree[int] = BinarySearchTree()
        for k in range(3000):   # three times the default recursion limit
            t.insert(k)
        self.assertEqual(t.height(), 2999)
        self.assertEqual(list(t), list(range(3000)))
        self.assertEqual(next(iter(t.postorder())), 2999)
        self.assertEqual(sum(1 for _ in t.level_order()), 3000)


if __name__ == "__main__":
    unittest.main()
