"""Run each program over the shared case files and compare its output with
tests/expected. verdicts.txt and user-verdicts.txt are shared with the C++ and
JavaScript examples: all three validators must agree on every case."""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
CASES = ROOT / "cases"
EXPECTED = ROOT / "tests" / "expected"


def cases(folder, prefix=""):
    return [f"{folder}/{p.name}" for p in sorted((CASES / folder).glob(f"{prefix}*.json"))]


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=CASES, capture_output=True, encoding="utf-8")
    assert r.returncode in (0, 1), r.stderr
    return r.stdout


def verdicts(out):
    found = (re.match(r"^([^ :]+\.json): (valid|invalid)", line) for line in out.splitlines())
    return "".join(f"{m[1]}: {m[2]}\n" for m in found if m)


def expected(name):
    return (EXPECTED / name).read_text(encoding="utf-8")


SYNTAX = cases("valid") + cases("invalid")
USERS = cases("schema", "user-")


class Outputs(unittest.TestCase):
    maxDiff = None

    def test_syntax_verdicts(self):
        self.assertEqual(verdicts(run(ROOT / "validate_json.py", *SYNTAX)), expected("verdicts.txt"))

    def test_syntax_messages(self):
        self.assertEqual(run(ROOT / "validate_json.py", *SYNTAX), expected("validate_json.txt"))

    def test_user_verdicts(self):
        self.assertEqual(verdicts(run(ROOT / "validate_user.py", *USERS)), expected("user-verdicts.txt"))

    def test_user_messages(self):
        self.assertEqual(run(ROOT / "validate_user.py", *USERS), expected("validate_user.txt"))

    def test_edge_cases(self):
        self.assertEqual(run(ROOT / "edge_cases.py", *cases("edge")), expected("edge_cases.txt"))

    def test_json_tool_reports_the_error(self):
        r = subprocess.run([sys.executable, "-m", "json.tool", "invalid/trailing-comma.json"],
                           cwd=CASES, capture_output=True, encoding="utf-8")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stderr.strip(),
                         "Illegal trailing comma before end of object: line 1 column 16 (char 15)")


if __name__ == "__main__":
    unittest.main()
