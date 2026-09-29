"""Test one number for primality, and list primes with the Sieve of Eratosthenes.

Run:  python3 primes.py
"""
from math import isqrt


def is_prime(n: int) -> bool:
    """Trial division by 2, 3 and then 6k - 1 and 6k + 1 up to sqrt(n)."""
    if n < 2:
        return False
    if n < 4:
        return True                      # 2 and 3
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, isqrt(n) + 1, 6):  # isqrt is exact for any int
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def sieve(limit: int) -> bytearray:
    """flags[k] is 1 when k is prime, for 0 <= k < limit."""
    flags = bytearray([1]) * limit
    flags[:2] = bytearray(min(limit, 2))  # 0 and 1 are not prime
    for p in range(2, isqrt(max(limit - 1, 0)) + 1):
        if flags[p]:
            flags[p * p::p] = bytearray(len(range(p * p, limit, p)))
    return flags


def count_primes_below(limit: int) -> int:
    return sum(sieve(limit))


def main() -> None:
    flags = sieve(100)
    print("primes below 100:", " ".join(str(k) for k in range(100) if flags[k]))
    print("primes below 1000:", count_primes_below(1000))
    print("primes below 1000000:", count_primes_below(1_000_000))
    samples = [-7, 0, 1, 2, 91, 97]
    print("is_prime:", ", ".join(f"{n} {str(is_prime(n)).lower()}" for n in samples))
    print("is_prime(2147483647) =", str(is_prime(2147483647)).lower())
    print("is_prime(1000000007) =", str(is_prime(1000000007)).lower())


if __name__ == "__main__":
    main()
