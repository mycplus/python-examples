# MD5, SHA-1 and SHA-256 in Python

[![hash-functions](https://github.com/mycplus/python-examples/actions/workflows/hash-functions.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/hash-functions.yml)

Companion code for [MD5, SHA-1 and SHA-256: Hash Functions in C, C++, Java, Python and C#](https://www.mycplus.com/computer-science/algorithms/md5-sha1-sha256-hash-functions/) on MYCPLUS: the three hashes and HMAC through `hashlib` and `hmac`, tested against the same vectors as the C and C++ implementations.

> For learning and testing. Never store passwords with a plain hash, and never build your own cryptography for real data: use a maintained library (OpenSSL, the platform's own crypto API) and, for passwords, a slow password hash such as Argon2id, scrypt or bcrypt.

| File | What it is |
| --- | --- |
| `digest_demo.py` | Hashes three short messages; prints exactly what the C version prints |
| `tests/test_digests.py` | Known answers, 306 message lengths, every split offset up to 130 bytes, HMAC-MD5/SHA-1/SHA-256, `hmac.compare_digest`, demo output |

The vectors are in `tests/vectors/`: the RFC 1321 and FIPS 180-4 examples, 306 message lengths generated with Python's `hashlib` (an independent implementation), and the RFC 4231 and RFC 2202 HMAC cases. `tests/vectors/` is identical in all five languages.

## Run

```sh
python -m unittest discover -s tests -v
python digest_demo.py
```

## What the build checks

Runs on Python 3.10 to 3.14 on Linux and Windows; `ruff check` and `mypy --strict` are clean; every vector passes and the demo prints exactly `tests/expected/digest_demo.txt`.
