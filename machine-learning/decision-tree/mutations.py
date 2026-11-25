"""Re-apply plausible mistakes to the tree code and check the tests catch each.

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
        "Gini without the 1 -",
        "tree.py",
        "1.0 - (p * p).sum(axis=-1)",
        "(p * p).sum(axis=-1)",
    ),
    ("entropy in natural log", "tree.py", "p * np.log2(p)", "p * np.log(p)"),
    (
        "right counts off by one row",
        "tree.py",
        "left[-1] + onehot[-1] - left",
        "left[-1] - left",
    ),
    (
        "splits between equal values allowed",
        "tree.py",
        "(x[1:] > x[:-1]) & ",
        "(x[1:] >= x[:-1]) & ",
    ),
    ("min_samples_leaf ignored", "tree.py", "(n_left >= min_leaf) & ", ""),
    (
        "children weighted equally",
        "tree.py",
        "n_left * impurity(left, criterion)\n            + (n - n_left) * impurity(right, criterion)\n        ) / n",
        "impurity(left, criterion)\n            + impurity(right, criterion)\n        ) / 2",
    ),
    (
        "ties go to the last feature",
        "tree.py",
        "gain[i] > best.gain",
        "gain[i] >= best.gain",
    ),
    ("threshold at the left value", "tree.py", "x[i] / 2.0 + x[i + 1] / 2.0", "x[i]"),
    (
        "depth limit one level too deep",
        "tree.py",
        "depth >= self.max_depth",
        "depth > self.max_depth",
    ),
    (
        "min_samples_split ignored",
        "tree.py",
        "or len(rows) < self.min_samples_split\n",
        "",
    ),
    ("pure nodes split again", "tree.py", "or (c > 0).sum() == 1  # pure", "or False"),
    (
        "training rows routed with <",
        "tree.py",
        "X_arr[rows, split.feature] <= split.threshold",
        "X_arr[rows, split.feature] < split.threshold",
    ),
    (
        "prediction routed with <",
        "tree.py",
        "Q[rows, self.feature_[n]] <= self.threshold_[n]",
        "Q[rows, self.feature_[n]] < self.threshold_[n]",
    ),
    (
        "children swapped",
        "tree.py",
        "np.where(go_left, self.left_[n], self.right_[n])",
        "np.where(go_left, self.right_[n], self.left_[n])",
    ),
    (
        "thresholds exported with 6 digits",
        "export.py",
        "{float(tree.threshold_[i])!r}",
        "{tree.threshold_[i]:.6g}",
    ),
    ("exported comparison uses <", "export.py", "<= {t!r}) {{", "< {t!r}) {{"),
    ("loader compares with <", "predict_exported.py", "row[f] <= t", "row[f] < t"),
]


def run(description: str, file: str, old: str, new: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "tree"
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
