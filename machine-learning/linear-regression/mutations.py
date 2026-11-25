"""Re-apply plausible mistakes to linear_regression.py and check the tests catch each.

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
    ("gradient sign flipped", "theta = theta - lr * grad", "theta = theta + lr * grad"),
    (
        "factor 2 dropped from the gradient",
        "(2.0 / len(y)) * (Xb.T",
        "(1.0 / len(y)) * (Xb.T",
    ),
    (
        "step from the smallest eigenvalue",
        "eigvalsh(hessian)[-1]",
        "eigvalsh(hessian)[0]",
    ),
    ("no intercept column", "np.column_stack([np.ones(X.shape[0]), X])", "X"),
    (
        "intercept dropped in predict",
        "X_arr @ self.coef_ + self.intercept_",
        "X_arr @ self.coef_",
    ),
    (
        "lstsq on X without the ones",
        "np.linalg.lstsq(Xb, y_arr)",
        "np.linalg.lstsq(add_intercept(X_arr[:, 1:]), y_arr)",
    ),
    (
        "MSE not divided by n",
        "float(residuals @ residuals) / len(y)",
        "float(residuals @ residuals)",
    ),
    (
        "stopping rule never relative",
        "stop_at = self.tol * float(np.linalg.norm(grad))",
        "stop_at = self.tol",
    ),
    ("divergence check removed", "if not np.isfinite(loss):", "if False:"),
    (
        "NaN check removed",
        "if not (np.isfinite(X_arr).all() and np.isfinite(y_arr).all()):",
        "if False:",
    ),
    ("coef_ keeps the intercept", "self.coef_ = theta[1:]", "self.coef_ = theta"),
]


def run(description: str, old: str, new: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "lr"
        shutil.copytree(
            HERE, work, ignore=shutil.ignore_patterns("__pycache__", "results")
        )
        source = work / "linear_regression.py"
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
