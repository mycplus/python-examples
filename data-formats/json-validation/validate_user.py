"""Validate user records against cases/schema/user.schema.json.

Usage: python validate_user.py FILE...
Needs: pip install jsonschema
"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = json.loads((Path(__file__).parent / "cases/schema/user.schema.json").read_text(encoding="utf-8"))
Draft202012Validator.check_schema(SCHEMA)  # fail early if the schema itself is wrong
validator = Draft202012Validator(SCHEMA)


def user_errors(data):
    """Return a list of "path: message" strings; empty when the record is valid."""
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    return [f"/{'/'.join(map(str, e.path))}: {e.message}" for e in errors]


def main(paths):
    invalid = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                errors = user_errors(json.load(f))
        except json.JSONDecodeError:
            errors = ["not valid JSON"]
        print(f"{path}: {'valid' if not errors else 'invalid'}")
        for e in errors:
            print(f"  {e}")
        invalid += bool(errors)
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
