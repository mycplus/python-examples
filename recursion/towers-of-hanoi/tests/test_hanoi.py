"""Run from the example directory:  python3 -m unittest discover -s tests"""
from __future__ import annotations

import io
import pathlib
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

# Run from the example directory, which puts hanoi.py on sys.path.
from hanoi import Move, main, move_at, move_count, solve, solve_iterative

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
EXPECTED = (HERE / "expected" / "towers_of_hanoi.txt").read_text()


def legal_and_solved(n: int, moves: list[Move]) -> bool:
    """Apply the moves to three pegs; False on any illegal move."""
    pegs: dict[str, list[int]] = {"A": list(range(n, 0, -1)), "B": [], "C": []}
    for m in moves:
        if m.source not in pegs or m.target not in pegs or m.source == m.target:
            return False
        src, dst = pegs[m.source], pegs[m.target]
        if not src or src[-1] != m.disk or (dst and dst[-1] < m.disk):
            return False
        dst.append(src.pop())
    return pegs == {"A": [], "B": [], "C": list(range(n, 0, -1))}


def reference_move(n: int, k: int, s: str, t: str, v: str) -> Move:
    """Move k found by descending the recursion: no bit tricks."""
    while True:
        mid = 1 << (n - 1)
        if k == mid:
            return Move(n, s, t)
        if k < mid:
            t, v = v, t
        else:
            k -= mid
            s, v = v, s
        n -= 1


class TestHanoi(unittest.TestCase):
    def test_solvers_agree_and_are_legal(self) -> None:
        for n in range(16):
            rec = list(solve(n))
            it = list(solve_iterative(n))
            self.assertTrue(legal_and_solved(n, rec), f"n={n}")
            self.assertEqual(rec, it, f"n={n}")
            self.assertEqual(len(rec), move_count(n))
            self.assertEqual([move_at(n, k) for k in range(1, len(rec) + 1)], rec)

    def test_other_pegs(self) -> None:
        moves = list(solve(7, "C", "A", "B"))
        swap = {"A": "C", "B": "B", "C": "A"}
        relabelled = [Move(m.disk, swap[m.source], swap[m.target]) for m in moves]
        self.assertTrue(legal_and_solved(7, relabelled))

    def test_move_count(self) -> None:
        self.assertEqual(move_count(0), 0)
        self.assertEqual(move_count(64), 18_446_744_073_709_551_615)
        self.assertEqual(move_count(100), 2**100 - 1)
        with self.assertRaises(ValueError):
            move_count(-1)

    def test_move_at_large_n_against_reference(self) -> None:
        rng = random.Random(20261006)
        for n in (64, 65, 100):
            last = move_count(n)
            ks = [1, last, 1 << (n - 1)] + [last + 1 - (1 << j) for j in range(n)]
            ks += [rng.randrange(1, last + 1) for _ in range(500)]
            for k in ks:
                self.assertEqual(move_at(n, k), reference_move(n, k, "A", "C", "B"),
                                 f"n={n} k={k}")

    def test_bad_input(self) -> None:
        for bad in ((-1, 1), (3, 0), (3, 8), (0, 1)):
            with self.assertRaises(ValueError):
                move_at(*bad)
        with self.assertRaises(ValueError):
            list(solve(-1))
        for arg in ("abc", "-1", "21", "3x", "2.5", "", " 3", "+3", "٣"):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(["hanoi.py", arg])
            self.assertEqual(status, 2, repr(arg))

    def test_outputs(self) -> None:
        for args in (["towers_of_hanoi.py"], ["hanoi.py", "3"], ["hanoi.py", "3", "--iterative"]):
            out = subprocess.run([sys.executable, *args], cwd=EXAMPLE_DIR, check=True,
                                 capture_output=True, text=True).stdout
            self.assertEqual(out.replace("\r", ""), EXPECTED, args)


if __name__ == "__main__":
    unittest.main()
