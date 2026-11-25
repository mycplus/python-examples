"""Re-apply plausible mistakes to logistic_regression.py and check the tests catch each.

Each mutant is written to a temporary copy of this folder; the original file
is never modified. A mutant that hangs counts as caught.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

MUTANTS = [
    (
        "sigmoid sign flipped",
        "1.0 / (1.0 + np.exp(-z[pos]))",
        "1.0 / (1.0 + np.exp(z[pos]))",
    ),
    (
        "gradient not divided by n",
        "Xb.T @ (sigmoid(Xb @ theta) - y) / len(y)",
        "Xb.T @ (sigmoid(Xb @ theta) - y)",
    ),
    (
        "prediction and label swapped",
        "(sigmoid(Xb @ theta) - y)",
        "(y - sigmoid(Xb @ theta))",
    ),
    (
        "intercept penalized in the gradient",
        "g[1:] += l2 * theta[1:]",
        "g += l2 * theta",
    ),
    ("Hessian without p(1 - p)", "(p * (1.0 - p))[:, None]", "p[:, None]"),
    (
        "Hessian without the penalty",
        "h[1:, 1:] += l2 * np.eye",
        "h[1:, 1:] += 0.0 * np.eye",
    ),
    (
        "curvature bound without the 1/4",
        "largest / (4.0 * float(Xb.shape[0]))",
        "largest / float(Xb.shape[0])",
    ),
    (
        "log loss sign error",
        "np.logaddexp(0.0, z) - y * z",
        "np.logaddexp(0.0, z) + y * z",
    ),
    ("Newton step added", "theta = theta - step", "theta = theta + step"),
    ("label check removed", "if not np.isin(y_arr, (0.0, 1.0)).all():", "if False:"),
    ("coef_ keeps the intercept", "self.coef_ = theta[1:]", "self.coef_ = theta"),
]


def run(description: str, old: str, new: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "lr"
        shutil.copytree(
            HERE, work, ignore=shutil.ignore_patterns("__pycache__", "results")
        )
        source = work / "logistic_regression.py"
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
for description, old, new in MUTANTS:
    killed = run(description, old, new)
    caught += killed
    print(f"{'caught' if killed else 'MISSED':6}  {description}")
print(f"\n{caught} of {len(MUTANTS)} mutants caught")
sys.exit(0 if caught == len(MUTANTS) else 1)
