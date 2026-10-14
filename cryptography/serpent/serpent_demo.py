"""Encrypt and decrypt one block with a 256-bit key, then check two published
test vectors: one in NESSIE byte order, and one from the Serpent AES
submission, which writes every value byte-reversed.

Run: python3 serpent_demo.py
"""
from __future__ import annotations

from serpent import Serpent


def main() -> None:
    key = bytes(range(32))  # 00 01 02 ... 1f
    plain = b"Serpent: 128-bit"  # 16 bytes
    cipher = Serpent(key)
    encrypted = cipher.encrypt(plain)
    back = cipher.decrypt(encrypted)

    print(f"{'key':<11}{key.hex()}")
    print(f"{'plaintext':<11}{plain.hex()}")
    print(f"{'ciphertext':<11}{encrypted.hex()}")
    print("decrypts back:", "yes" if back == plain else "no")

    # NESSIE: 128-bit key 80 00 .. 00, zero block -> 264e5481...
    nessie = Serpent(b"\x80" + bytes(15)).encrypt(bytes(16))
    ok = nessie == bytes.fromhex("264e5481eff42a4606abda06c0bfda3d")
    print("NESSIE vector:         ", "match" if ok else "MISMATCH")

    # AES submission: 256-bit zero key, zero block -> 89104945... once the
    # output bytes are reversed (the key and block are all zero either way).
    out = Serpent(bytes(32)).encrypt(bytes(16))[::-1]
    ok = out == bytes.fromhex("8910494504181950f98dd998a82b6749")
    print("AES submission vector: ", "match" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
