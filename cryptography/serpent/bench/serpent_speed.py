"""Encryption speed in ECB mode over a buffer of N MiB (default 1; pure
Python is slow), median of 5 runs. Every language's benchmark fills the
buffer the same way and prints the XOR of all ciphertext blocks.

Usage: python3 bench/serpent_speed.py [MiB]
"""
from __future__ import annotations

import pathlib
import platform
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from serpent import Serpent


def main() -> None:
    mib = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    size = mib << 20
    data = bytes(((i * 2654435761) & 0xFFFFFFFF) >> 24 for i in range(size))
    cipher = Serpent(bytes(range(32)))
    blocks = [data[i : i + 16] for i in range(0, size, 16)]

    rates = []
    out: list[bytes] = []
    for _ in range(5):
        start = time.perf_counter()
        out = [cipher.encrypt(b) for b in blocks]
        rates.append(size / 1e6 / (time.perf_counter() - start))
    fold = bytearray(16)
    for block in out:
        for i in range(16):
            fold[i] ^= block[i]
    label = f"Python {platform.python_version()}"
    print(f"{label:<12} {statistics.median(rates):8.3f} MB/s (median of 5, {mib} MiB)"
          f"  xor of blocks {fold.hex()}")


if __name__ == "__main__":
    main()
