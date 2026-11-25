"""Re-apply plausible mistakes to knn.py and data.py and check the tests catch each.

Each mutant is written to a temporary copy of this folder; the original files
are never modified. A mutant that hangs counts as caught.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (description, file, original text, mutated text)
MUTANTS = [
    (
        "square root left out of the distance",
        "knn.py",
        'np.sqrt(np.einsum("qnd,qnd->qn", diff, diff))',
        'np.einsum("qnd,qnd->qn", diff, diff)',
    ),
    (
        "manhattan distance without abs()",
        "knn.py",
        "np.abs(diff).sum(axis=2)",
        "diff.sum(axis=2)",
    ),
    (
        "neighbors picked with argpartition, never sorted",
        "knn.py",
        'order = np.argsort(distances, axis=1, kind="stable")',
        "order = np.argpartition(distances, k - 1, axis=1)",
    ),
    ("one neighbor too few", "knn.py", "order[:, :k]", "order[:, : max(k - 1, 1)]"),
    ("query counted as neighbor 0", "knn.py", "order[:, :k]", "order[:, 1 : k + 1]"),
    ("weights 1/d squared", "knn.py", "w = 1.0 / distances", "w = 1.0 / distances**2"),
    (
        "zero-distance rule dropped",
        "knn.py",
        "w[has_exact] = exact[has_exact].astype(np.float64)",
        "pass",
    ),
    (
        "tied vote goes to the largest label",
        "knn.py",
        "winners = np.argmax(self.predict_proba(X), axis=1)",
        (
            "p = self.predict_proba(X)\n"
            "        winners = p.shape[1] - 1 - np.argmax(p[:, ::-1], axis=1)"
        ),
    ),
    (
        "k larger than the data accepted",
        "knn.py",
        "if self.k > X_arr.shape[0]:",
        "if False:",
    ),
    ("NaN accepted", "knn.py", "if not np.isfinite(X_arr).all():", "if False:"),
    (
        "labels assumed to be 0, 1, 2, ...",
        "knn.py",
        "self.classes_, self.y_codes_ = np.unique(y_arr, return_inverse=True)",
        "self.classes_, self.y_codes_ = np.unique(y_arr), y_arr",
    ),
    (
        "sample standard deviation (ddof=1)",
        "data.py",
        "std = X_train.std(axis=0)",
        "std = X_train.std(axis=0, ddof=1)",
    ),
    (
        "test rows scaled with their own statistics",
        "data.py",
        "return (X_train - mean) / std, (X_test - mean) / std",
        "return (X_train - mean) / std, (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)",
    ),
    (
        "mean not subtracted",
        "data.py",
        "return (X_train - mean) / std, (X_test - mean) / std",
        "return X_train / std, X_test / std",
    ),
    (
        "splitmix64 increment wrong",
        "data.py",
        "0x9E3779B97F4A7C15",
        "0x9E3779B97F4A7C17",
    ),
    (
        "uniform from the low bits",
        "data.py",
        "(self.next() >> 11) * 2.0**-53",
        "(self.next() & ((1 << 53) - 1)) * 2.0**-53",
    ),
]


def run(description: str, file: str, old: str, new: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "knn"
        shutil.copytree(
            HERE,
            work,
            ignore=shutil.ignore_patterns("__pycache__", "results", ".*cache"),
        )
        source = work / file
        text = source.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"mutant '{description}': pattern not found exactly once")
        source.write_text(text.replace(old, new))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                check=False,
                cwd=work,
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return True
        return result.returncode != 0


caught = 0
for description, file, old, new in MUTANTS:
    killed = run(description, file, old, new)
    caught += killed
    print(f"{'caught' if killed else 'MISSED':6}  {description}")
print(f"\n{caught} of {len(MUTANTS)} mutants caught")
sys.exit(0 if caught == len(MUTANTS) else 1)
