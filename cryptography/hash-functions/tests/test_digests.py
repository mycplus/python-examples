"""Checks hashlib and hmac against the published and independently generated answers,
one-shot, in chunks and split at every offset. Run: python -m unittest discover -s tests -v
"""
import hashlib
import hmac
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import digest_demo

VECTORS = Path(__file__).resolve().parent / "vectors"
NAMES = ("md5", "sha1", "sha256")


def rows(name: str) -> list[list[str]]:
    with open(VECTORS / name, encoding="utf-8", newline="") as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    return [ln.split("\t") for ln in lines if ln and not ln.startswith("#")]


def pattern(n: int) -> bytes:
    return bytes((i * 31 + 7) & 0xFF for i in range(n))


class DigestTests(unittest.TestCase):
    def test_known_answers(self) -> None:
        count = 0
        for algo, repeat, text, want in rows("known_answers.txt"):
            message = (text * int(repeat)).encode()
            with self.subTest(algo=algo, text=text[:20]):
                self.assertEqual(digest_demo.digest_hex(algo, message), want)
                h = hashlib.new(algo)
                for i in range(0, len(message), 7):          # 7-byte chunks
                    h.update(message[i : i + 7])
                self.assertEqual(h.hexdigest(), want)
            count += 1
        self.assertEqual(count, 16)

    def test_lengths_and_splits(self) -> None:
        count = 0
        for row in rows("lengths.txt"):
            n = int(row[0])
            msg = pattern(n)
            for algo, want in zip(NAMES, row[1:]):
                with self.subTest(algo=algo, length=n):
                    self.assertEqual(digest_demo.digest_hex(algo, msg), want)
                    if n <= 130:                              # split at every offset
                        for cut in range(n + 1):
                            h = hashlib.new(algo)
                            h.update(msg[:cut])
                            h.update(msg[cut:])
                            self.assertEqual(h.hexdigest(), want)
            count += 1
        self.assertEqual(count, 306)

    def test_hmac_vectors(self) -> None:
        count = 0
        for algo, key, data, want in rows("hmac.txt"):
            with self.subTest(algo=algo, key=key[:16]):
                got = hmac.new(bytes.fromhex(key), bytes.fromhex(data), algo).hexdigest()
                self.assertEqual(got, want)
                self.assertTrue(digest_demo.tags_match(got, want))
            count += 1
        self.assertEqual(count, 10)

    def test_tags_match_rejects_a_changed_digest(self) -> None:
        good = digest_demo.digest_hex("sha256", b"abc")
        self.assertTrue(digest_demo.tags_match(good, good))
        self.assertFalse(digest_demo.tags_match(good, good[:-1] + "0"))

    def test_demo_prints_the_article_output(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            digest_demo.main()
        expected = (Path(__file__).resolve().parent / "expected" / "digest_demo.txt").read_text(encoding="utf-8")
        self.assertEqual(out.getvalue().replace("\r\n", "\n"), expected.replace("\r\n", "\n"))


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main()
