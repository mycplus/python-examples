"""Towers of Hanoi: recursive and iterative solvers, any single move
computed directly, and the move count.

Run:  python3 hanoi.py N [--iterative]     (0 <= N <= 20)
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from typing import NamedTuple


class Move(NamedTuple):
    disk: int      # 1 is the smallest
    source: str
    target: str


def move_count(n: int) -> int:
    """2^n - 1. Python integers do not overflow, so any n >= 0 works."""
    if n < 0:
        raise ValueError(f"negative disk count: {n}")
    return (1 << n) - 1


def solve(n: int, source: str = "A", target: str = "C", spare: str = "B") -> Iterator[Move]:
    """Yield the moves of the recursive solution. Recursion depth is n + 1."""
    if n < 0:
        raise ValueError(f"negative disk count: {n}")
    if n == 0:
        return
    yield from solve(n - 1, source, spare, target)
    yield Move(n, source, target)
    yield from solve(n - 1, spare, target, source)


def move_at(n: int, k: int, source: str = "A", target: str = "C", spare: str = "B") -> Move:
    """Move k (1 <= k <= 2^n - 1) of the n-disk solution, from the bits of k."""
    if n < 1 or not 1 <= k <= move_count(n):
        raise ValueError(f"n or k out of range: n={n}, k={k}")
    # The formula moves the tower from peg index 0 to index 2 when n is odd
    # and to index 1 when n is even.
    peg = (source, spare, target) if n % 2 == 1 else (source, target, spare)
    disk = (k & -k).bit_length()           # 1 + number of trailing zero bits
    return Move(disk, peg[(k & (k - 1)) % 3], peg[((k | (k - 1)) + 1) % 3])


def solve_iterative(n: int, source: str = "A", target: str = "C",
                    spare: str = "B") -> Iterator[Move]:
    """Yield the same moves as solve(), with no recursion."""
    for k in range(1, move_count(n) + 1):
        yield move_at(n, k, source, target, spare)


def main(argv: list[str]) -> int:
    iterative = len(argv) == 3 and argv[2] == "--iterative"
    try:
        if len(argv) != 2 and not iterative:
            raise ValueError
        # ASCII digits only: int() alone would also accept " 3", "+3" and
        # non-ASCII digits such as "\u0663" (ARABIC-INDIC DIGIT THREE).
        if not re.fullmatch(r"[0-9]+", argv[1]):
            raise ValueError
        n = int(argv[1])
        if not 0 <= n <= 20:
            raise ValueError
    except ValueError:
        print(f"usage: {argv[0]} N [--iterative]   (N is 0 to 20)", file=sys.stderr)
        return 2
    moves = solve_iterative(n) if iterative else solve(n)
    for m in moves:
        print(f"Move disk {m.disk} from {m.source} to {m.target}")
    print(f"{n} disks: {move_count(n)} moves")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
