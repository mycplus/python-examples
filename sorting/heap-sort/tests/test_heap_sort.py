"""Run from the example directory:  python3 -m unittest discover -s tests"""
from __future__ import annotations

import io
import pathlib
import random
import runpy
import unittest
from contextlib import redirect_stdout

# Run from the example directory, which puts heap_sort.py on sys.path.
from heap_sort import heap_sort

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLE = HERE.parent / "heap_sort.py"


class TestHeapSort(unittest.TestCase):
    def test_matches_sorted(self) -> None:
        # 5,000 random lists of 0 to 64 values; every third one draws from
        # only 4 values, so equal keys are common.
        rng = random.Random(20261004)
        for case in range(5_000):
            n = rng.randrange(65)
            spread = 4 if case % 3 == 0 else 1_000
            data = [rng.randrange(-spread, spread) for _ in range(n)]
            original = list(data)
            result = heap_sort(data)
            self.assertEqual(result, sorted(original), f"case {case}")
            self.assertIs(result, data)  # sorted in place, same list returned

    def test_large_and_ordered_inputs(self) -> None:
        rng = random.Random(7)
        big = [rng.randrange(-10**9, 10**9) for _ in range(20_000)]
        self.assertEqual(heap_sort(list(big)), sorted(big))
        self.assertEqual(heap_sort(list(range(2_000, 0, -1))), list(range(1, 2_001)))
        self.assertEqual(heap_sort(list(range(2_000))), list(range(2_000)))

    def test_empty_and_single(self) -> None:
        self.assertEqual(heap_sort([]), [])
        self.assertEqual(heap_sort([42]), [42])

    def test_demo_output(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        expected = (HERE / "expected" / "heap_sort.txt").read_text(encoding="utf-8")
        self.assertEqual(buf.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
