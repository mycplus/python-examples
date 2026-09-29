# Prime numbers in Python

[![prime-numbers](https://github.com/mycplus/python-examples/actions/workflows/prime-numbers.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/prime-numbers.yml)

Companion code for [Prime Number Programs in C, C++, Java, Python, C#, PHP and JavaScript](https://www.mycplus.com/computer-science/algorithms/prime-number-program/) on MYCPLUS: a primality test by trial division up to the square root, and the Sieve of Eratosthenes. The same program exists in seven languages and every version prints the same output.

| File | What it is |
| --- | --- |
| `primes.py` | `is_prime()`, a `bytearray` sieve and the demo |
| `miller_rabin.py` | A deterministic Miller–Rabin test for integers below 2^64 |
| `tests/test_primes.py` | `unittest` tests |

Requires Python 3.10 or later; no third-party packages at run time.

## Build and test

```sh
python3 primes.py
python3 -m unittest discover -s tests -v
```

## What the build checks

- Runs on Python 3.10 to 3.14, on Linux and Windows; `ruff` 0.16.9 and `mypy --strict` 2.3.1 report nothing.
- Trial division agrees with the sieve on every number below 200,000.
- Known primes (including 2147483647, 1000000007 and 999999999989) and composites (including negatives, 0, 1, squares of primes and Carmichael numbers 561 and 1105) are classified correctly.
- The sieve reproduces the published prime counts: 168 below 1,000, 78,498 below 1,000,000 and 664,579 below 10,000,000.
- Miller–Rabin agrees with trial division below 200,000, accepts the largest primes below 2^63 and 2^64, and rejects strong pseudoprimes to smaller base sets (3215031751, 3825123056546413051).
- The demo prints exactly `tests/expected/primes.txt`, the same file in all seven repositories.
