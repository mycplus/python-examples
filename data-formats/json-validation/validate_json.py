"""Report whether each file holds one valid JSON text (RFC 8259).

Usage: python validate_json.py FILE...
"""
import json
import sys


def reject_constant(name):
    # Python accepts NaN, Infinity and -Infinity by default; JSON does not.
    raise ValueError(f"{name} is not valid JSON")


def check_json(text):
    """Return None for valid JSON, otherwise the parser's message."""
    try:
        json.loads(text, parse_constant=reject_constant)
        return None
    except ValueError as e:
        # json.JSONDecodeError is a ValueError subclass with e.lineno, e.colno
        # and e.pos; reject_constant raises a plain ValueError.
        return str(e)
    except RecursionError:
        return "nested too deeply for this parser"


def main(paths):
    invalid = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            error = check_json(f.read())
        print(f"{path}: valid" if error is None else f"{path}: invalid ({error})")
        invalid += error is not None
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
