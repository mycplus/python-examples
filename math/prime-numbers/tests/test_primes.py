"""Run from the example directory:  python3 -m unittest discover -s tests"""
import io
import pathlib
import unittest
from contextlib import redirect_stdout

# Run from the example directory, which puts primes.py on sys.path.
import primes
from miller_rabin import is_prime_mr
from primes import count_primes_below, is_prime, sieve

HERE = pathlib.Path(__file__).resolve().parent


class PrimesTest(unittest.TestCase):
    def test_small_and_edge_values(self) -> None:
        for p in [2, 3, 5, 7, 11, 13, 97, 7919, 1000003,
                  2147483647, 1000000007, 999999999989]:
            self.assertTrue(is_prime(p), p)
        for c in [-2**63, -7, -1, 0, 1, 4, 6, 9, 25, 49, 91, 121, 561, 1105, 7917,
                  1000003 * 1000003, 1000003 * 1000033, 2147483647 * 2]:
            self.assertFalse(is_prime(c), c)

    def test_trial_division_matches_sieve(self) -> None:
        flags = sieve(200_000)
        for k in range(200_000):
            self.assertEqual(is_prime(k), bool(flags[k]), k)

    def test_prime_counts(self) -> None:
        for limit, count in [(0, 0), (1, 0), (2, 0), (3, 1), (10, 4), (100, 25),
                             (1000, 168), (1_000_000, 78498), (10_000_000, 664579)]:
            self.assertEqual(count_primes_below(limit), count, limit)

    def test_miller_rabin_matches_trial_division(self) -> None:
        for k in range(-10, 200_000):
            self.assertEqual(is_prime_mr(k), is_prime(k), k)
        for n in [2147483647, 1000000007, 999999999989, 9223372036854775783,
                  18446744073709551557]:                 # largest prime below 2**64
            self.assertTrue(is_prime_mr(n), n)
        for n in [3215031751, 3825123056546413051,       # strong pseudoprimes to small bases
                  1000003 * 1000033, 4294967291 * 4294967279]:
            self.assertFalse(is_prime_mr(n), n)
        with self.assertRaises(ValueError):
            is_prime_mr(2**64 + 13)

    def test_demo_output(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            primes.main()
        expected = (HERE / "expected" / "primes.txt").read_text()
        self.assertEqual(buf.getvalue(), expected.replace("\r", ""))


if __name__ == "__main__":
    unittest.main()
