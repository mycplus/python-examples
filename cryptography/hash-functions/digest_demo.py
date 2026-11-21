"""MD5, SHA-1 and SHA-256 of three short messages, with hashlib.

Run: python digest_demo.py
"""
import hashlib
import hmac


def digest_hex(algorithm: str, data: bytes) -> str:
    """Hex digest of data for a hashlib algorithm name such as "sha256"."""
    return hashlib.new(algorithm, data).hexdigest()


def tags_match(expected: str, received: str) -> bool:
    """Compare two hex digests in constant time, as a real verifier must."""
    return hmac.compare_digest(expected, received)


def show(text: str) -> None:
    data = text.encode("utf-8")
    print(f'message: "{text}" ({len(data)} bytes)')
    print(f"  MD5     {digest_hex('md5', data)}")
    print(f"  SHA-1   {digest_hex('sha1', data)}")
    print(f"  SHA-256 {digest_hex('sha256', data)}")


def main() -> None:
    show("")
    show("abc")
    show("The quick brown fox jumps over the lazy dog")


if __name__ == "__main__":
    main()
