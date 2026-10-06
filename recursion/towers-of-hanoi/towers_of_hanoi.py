"""The recursive Towers of Hanoi solution in Python 3.

Run:  python3 towers_of_hanoi.py
"""


def hanoi(n: int, source: str, target: str, spare: str) -> int:
    """Move n disks from source to target, using spare as the third peg.

    Prints each move and returns the number of moves made.
    """
    if n <= 0:
        return 0                                    # nothing to move
    moves = hanoi(n - 1, source, spare, target)
    print(f"Move disk {n} from {source} to {target}")
    moves += 1
    moves += hanoi(n - 1, spare, target, source)
    return moves


if __name__ == "__main__":
    n = 3
    moves = hanoi(n, "A", "C", "B")
    print(f"{n} disks: {moves} moves")
