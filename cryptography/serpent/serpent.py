"""The Serpent block cipher (128-bit blocks, keys up to 256 bits).

Bytes are read as little-endian 32-bit words, the order the NESSIE test
vectors and libgcrypt use.

For learning and testing. It has not been reviewed for side channels, and
pure Python is slow; use a maintained cryptographic library for anything
that protects real data.
"""
from __future__ import annotations

BLOCK_SIZE = 16
MAX_KEY_SIZE = 32
MASK = 0xFFFFFFFF
PHI = 0x9E3779B9  # fractional part of the golden ratio

SBOX: tuple[tuple[int, ...], ...] = (
    (3, 8, 15, 1, 10, 6, 5, 11, 14, 13, 4, 2, 7, 0, 9, 12),
    (15, 12, 2, 7, 9, 0, 5, 10, 1, 11, 14, 8, 6, 13, 3, 4),
    (8, 6, 7, 9, 3, 12, 10, 15, 13, 1, 14, 4, 0, 11, 5, 2),
    (0, 15, 11, 8, 12, 9, 6, 3, 13, 1, 2, 4, 10, 7, 5, 14),
    (1, 15, 8, 3, 12, 0, 11, 6, 2, 5, 4, 10, 9, 14, 7, 13),
    (15, 5, 2, 11, 4, 10, 9, 12, 0, 3, 14, 8, 13, 6, 7, 1),
    (7, 2, 12, 5, 8, 4, 6, 11, 14, 9, 1, 15, 13, 3, 10, 0),
    (1, 13, 15, 0, 14, 8, 2, 11, 7, 4, 12, 10, 9, 3, 5, 6),
)
# Inverse S-boxes, computed from SBOX so there is no second table to mistype.
SBOX_INV: tuple[tuple[int, ...], ...] = tuple(
    tuple(box.index(v) for v in range(16)) for box in SBOX
)

Words = list[int]


def _rotl(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & MASK


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK


def _apply_sbox(box: tuple[int, ...], x: Words) -> Words:
    """Bit j of x[0..3] forms a 4-bit input (x[0] is the low bit)."""
    y = [0, 0, 0, 0]
    for j in range(32):
        nibble = 0
        for b in range(4):
            nibble |= ((x[b] >> j) & 1) << b
        out = box[nibble]
        for b in range(4):
            y[b] |= ((out >> b) & 1) << j
    return y


def _linear_transform(x: Words) -> Words:
    x0, x1, x2, x3 = x
    x0 = _rotl(x0, 13)
    x2 = _rotl(x2, 3)
    x1 ^= x0 ^ x2
    x3 ^= x2 ^ ((x0 << 3) & MASK)
    x1 = _rotl(x1, 1)
    x3 = _rotl(x3, 7)
    x0 ^= x1 ^ x3
    x2 ^= x3 ^ ((x1 << 7) & MASK)
    x0 = _rotl(x0, 5)
    x2 = _rotl(x2, 22)
    return [x0, x1, x2, x3]


def _inverse_linear_transform(x: Words) -> Words:
    x0, x1, x2, x3 = x
    x2 = _rotr(x2, 22)
    x0 = _rotr(x0, 5)
    x2 ^= x3 ^ ((x1 << 7) & MASK)
    x0 ^= x1 ^ x3
    x3 = _rotr(x3, 7)
    x1 = _rotr(x1, 1)
    x3 ^= x2 ^ ((x0 << 3) & MASK)
    x1 ^= x0 ^ x2
    x2 = _rotr(x2, 3)
    x0 = _rotr(x0, 13)
    return [x0, x1, x2, x3]


def _load(block: bytes) -> Words:
    if len(block) != BLOCK_SIZE:
        raise ValueError("Serpent blocks are 16 bytes")
    return [int.from_bytes(block[i : i + 4], "little") for i in range(0, 16, 4)]


def _store(x: Words) -> bytes:
    return b"".join(w.to_bytes(4, "little") for w in x)


class Serpent:
    """Serpent with a key of 1 to 32 bytes; encrypts and decrypts 16-byte blocks."""

    def __init__(self, key: bytes) -> None:
        if not 1 <= len(key) <= MAX_KEY_SIZE:
            raise ValueError("Serpent keys are 1 to 32 bytes")
        # Pad to 256 bits: one 1 bit after the key, then zeros.
        padded = bytes(key) + (b"\x01" if len(key) < MAX_KEY_SIZE else b"")
        padded = padded.ljust(MAX_KEY_SIZE, b"\x00")
        # w[0..7] hold the padded key; w[8..139] are the 132 prekey words.
        w = [int.from_bytes(padded[i : i + 4], "little") for i in range(0, 32, 4)]
        for i in range(8, 140):
            w.append(_rotl(w[i - 8] ^ w[i - 5] ^ w[i - 3] ^ w[i - 1] ^ PHI ^ (i - 8), 11))
        # Subkey i passes prekeys 4i..4i+3 through S-box (3 - i) mod 8.
        self._subkeys: list[Words] = [
            _apply_sbox(SBOX[(3 - i) % 8], w[8 + 4 * i : 12 + 4 * i]) for i in range(33)
        ]

    def encrypt(self, block: bytes) -> bytes:
        x = _load(block)
        for r in range(32):
            x = [a ^ k for a, k in zip(x, self._subkeys[r])]
            x = _apply_sbox(SBOX[r % 8], x)
            if r < 31:
                x = _linear_transform(x)
            else:  # the last round replaces LT with K32
                x = [a ^ k for a, k in zip(x, self._subkeys[32])]
        return _store(x)

    def decrypt(self, block: bytes) -> bytes:
        x = _load(block)
        for r in range(31, -1, -1):
            if r == 31:
                x = [a ^ k for a, k in zip(x, self._subkeys[32])]
            else:
                x = _inverse_linear_transform(x)
            x = _apply_sbox(SBOX_INV[r % 8], x)
            x = [a ^ k for a, k in zip(x, self._subkeys[r])]
        return _store(x)
