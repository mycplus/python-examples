"""Run from the example directory:  python3 -m unittest discover -s tests"""
from __future__ import annotations

import io
import pathlib
import random
import sys
import unittest
from contextlib import redirect_stdout
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import stack  # noqa: E402
from stack import ArrayStack, LinkedStack, StackUnderflowError, balanced  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


class Shared:
    """Holder class, so unittest does not collect Base on its own."""

    class Base(unittest.TestCase):
        make: Callable[[], Any]

        def test_matches_list_model(self) -> None:
            rng = random.Random(20260928)
            s = self.make()
            model: list[str] = []
            for _ in range(100_000):
                if rng.randrange(3):
                    v = f"value-{rng.randrange(100_000)}"
                    s.push(v)
                    model.append(v)
                elif model:
                    self.assertEqual(s.pop(), model.pop())
                else:
                    self.assertTrue(s.is_empty())
                self.assertEqual(len(s), len(model))
                if model:
                    self.assertEqual(s.peek(), model[-1])

        def test_empty_stack_raises(self) -> None:
            s = self.make()
            self.assertTrue(s.is_empty())
            for op in (s.pop, s.peek):
                with self.assertRaises(StackUnderflowError):
                    op()
            # Still an IndexError, like list.pop() on an empty list.
            with self.assertRaises(IndexError):
                s.pop()

        def test_none_is_a_value(self) -> None:
            s = self.make()
            s.push(None)
            self.assertEqual(len(s), 1)
            self.assertIsNone(s.pop())


class ArrayStackTest(Shared.Base):
    make = ArrayStack


class LinkedStackTest(Shared.Base):
    make = LinkedStack

    def test_deep_stack_deletes(self) -> None:
        s: LinkedStack[int] = LinkedStack()
        for i in range(1_000_000):
            s.push(i)
        self.assertEqual(s.peek(), 999_999)
        del s  # must not hit a recursion limit or crash


class BalancedTest(unittest.TestCase):
    def test_cases(self) -> None:
        for text, want in [("", True), ("()[]{}", True), ("{[()()]}", True),
                           ("a(b)c", True), ("(", False), (")", False),
                           ("([)]", False), ("((", False), ("())", False),
                           ("}{", False)]:
            with self.subTest(text=text):
                self.assertEqual(balanced(text), want)

    def test_deep_nesting(self) -> None:
        self.assertTrue(balanced("(" * 100_000 + ")" * 100_000))


class DemoOutputTest(unittest.TestCase):
    def test_matches_article(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            stack.main()
        expected = (HERE / "expected" / "stack.txt").read_text()
        self.assertEqual(buf.getvalue(), expected.replace("\r", ""))


if __name__ == "__main__":
    unittest.main()
