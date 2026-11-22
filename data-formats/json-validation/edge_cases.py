"""Show what Python's json module does, with its default settings, with
inputs that JSON parsers handle differently.

Usage: python edge_cases.py FILE...
"""
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        value = json.loads(text)
        # Very deep values are not printed.
        shown = "" if len(text) > 1000 else f", value {json.dumps(value, separators=(',', ':'))}"
        print(f"{path}: accepted{shown}")
    except (ValueError, RecursionError) as e:
        print(f"{path}: rejected ({type(e).__name__}: {e})")
