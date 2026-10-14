# Serpent Block Cipher in Python

[![serpent](https://github.com/mycplus/python-examples/actions/workflows/serpent.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/serpent.yml)

Companion code for [Serpent Cipher Explained](https://www.mycplus.com/computer-science/algorithms/serpent-cipher/) on MYCPLUS: a pure-Python implementation of the Serpent block cipher (128-bit blocks, keys up to 256 bits), checked against 1,043 published test vectors. No third-party packages.

> For learning and testing. This code has not been reviewed for side channels, and pure Python encrypts only about 20 KB per second. To protect real data, use a maintained library such as [`cryptography`](https://cryptography.io/) with AES-GCM or ChaCha20-Poly1305.

| File | What it is |
| --- | --- |
| `serpent.py` | `Serpent(key)` with `encrypt` and `decrypt` of one 16-byte block; the inverse S-boxes are computed from the forward ones |
| `serpent_demo.py` | Encrypts one block with a 256-bit key and checks two known vectors |
| `tests/test_serpent.py` | Every vector in `tests/vectors/serpent_nessie.txt` in both directions, key and block lengths, the demo's output |
| `bench/serpent_speed.py` | Throughput in ECB mode over 1 MiB |

The test vectors are the NESSIE set as distributed with [Botan](https://github.com/randombit/botan) (`src/tests/data/block/serpent.vec`).

## Build and run

Python 3.10 or later.

```sh
python3 -m unittest discover -s tests -v
python3 serpent_demo.py
python3 bench/serpent_speed.py
```

## What the build checks

- `ruff check` and `mypy --strict` pass, on Python 3.10 to 3.14, on Linux and Windows.
- All 1,043 vectors (128-, 192- and 256-bit keys) encrypt and decrypt correctly; keys of 0 and 33 bytes and a 15-byte block raise `ValueError`.
- The demo prints exactly `tests/expected/serpent_demo.txt`.
