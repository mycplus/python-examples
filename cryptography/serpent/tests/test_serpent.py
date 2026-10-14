"""Run from the example directory:  python3 -m unittest discover -s tests"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest
from collections import Counter

# Run from the example directory, which puts serpent.py on sys.path.
from serpent import SBOX, SBOX_INV, Serpent

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
VECTORS = HERE / "vectors" / "serpent_nessie.txt"
EXPECTED = (HERE / "expected" / "serpent_demo.txt").read_text()


def load_vectors() -> list[tuple[bytes, bytes, bytes]]:
    vectors = []
    for line in VECTORS.read_text().splitlines():
        fields = line.split()
        if line.startswith("#") or len(fields) != 3:
            continue
        key, plain, cipher = (bytes.fromhex(f) for f in fields)
        vectors.append((key, plain, cipher))
    return vectors


class TestSerpent(unittest.TestCase):
    def test_all_published_vectors(self) -> None:
        vectors = load_vectors()
        passed: Counter[int] = Counter()
        for key, plain, expected in vectors:
            s = Serpent(key)
            with self.subTest(key=key.hex(), plain=plain.hex()):
                self.assertEqual(s.encrypt(plain), expected)
                self.assertEqual(s.decrypt(expected), plain)
                passed[len(key) * 8] += 1
        self.assertEqual(len(vectors), 1043)
        self.assertEqual(passed, Counter({128: 258, 192: 327, 256: 458}))

    def test_inverse_sboxes_invert(self) -> None:
        for box, inv in zip(SBOX, SBOX_INV):
            self.assertEqual([inv[box[i]] for i in range(16)], list(range(16)))

    def test_key_and_block_lengths(self) -> None:
        for bad in (b"", bytes(33)):
            with self.assertRaises(ValueError):
                Serpent(bad)
        Serpent(b"\x00")  # a 1-byte key is legal: it is padded like any short key
        with self.assertRaises(ValueError):
            Serpent(bytes(16)).encrypt(bytes(15))

    def test_demo_prints_the_article_output(self) -> None:
        out = subprocess.run(
            [sys.executable, "serpent_demo.py"],
            cwd=EXAMPLE_DIR, capture_output=True, text=True, check=True,
        ).stdout
        self.assertEqual(out, EXPECTED)


if __name__ == "__main__":
    unittest.main()
