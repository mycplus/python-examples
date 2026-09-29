"""Deterministic Miller-Rabin test for integers below 2**64.

Testing the first twelve primes as bases is enough for every n < 2**64.
Run:  python3 miller_rabin.py
"""
from time import perf_counter

BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime_mr(n: int) -> bool:
    if n < 2:
        return False
    for p in BASES:
        if n % p == 0:
            return n == p
    if n >= 2**64:
        raise ValueError("these bases are proven only for n < 2**64")
    d, s = n - 1, 0
    while d % 2 == 0:                  # n - 1 = d * 2**s with d odd
        d //= 2
        s += 1
    for a in BASES:
        x = pow(a, d, n)               # modular exponentiation
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False               # a proves n composite
    return True


if __name__ == "__main__":
    n = 9223372036854775783            # the largest prime below 2**63
    start = perf_counter()
    result = is_prime_mr(n)
    print(f"is_prime_mr({n}) = {str(result).lower()} in {(perf_counter() - start) * 1e6:.0f} microseconds")
