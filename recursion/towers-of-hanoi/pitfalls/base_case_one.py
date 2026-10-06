"""DO NOT COPY. The base case is n == 1, so n == 0 never reaches it and the
recursion runs until Python's recursion limit stops it."""
import sys


def hanoi(n: int, source: str, target: str, spare: str) -> int:
    if n == 1:
        return 1
    return hanoi(n - 1, source, spare, target) + 1 + hanoi(n - 1, spare, target, source)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"solving for {n} disks", flush=True)
    print(f"{n} disks: {hanoi(n, 'A', 'C', 'B')} moves")
