"""What happens without regularization when the classes are separable.

The 455 standardized training rows can be split perfectly by a hyperplane, so
the unpenalized log loss has no minimum: it keeps falling as the weights grow.
"""

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from data import load_split
from logistic_regression import LogisticRegressionScratch

X_train, X_test, y_train, y_test, _ = load_split()

# Gradient descent with no penalty, stopped at increasing step counts.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    gd = LogisticRegressionScratch(method="gd", l2=0.0, max_iter=1_000_000).fit(
        X_train, y_train
    )
print("gradient descent, no penalty, on the training rows:")
print(f"{'steps':>9} {'train loss':>11} {'weight norm':>12}")
checkpoints = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
for k in checkpoints:
    print(f"{k:9,} {gd.loss_history_[k]:11.2e} {gd.weight_norm_history_[k]:12.1f}")
print(f"converged: {gd.converged_}")


def report(label: str, p: np.ndarray, norm: float) -> dict[str, float]:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    row = {
        "test_acc": accuracy_score(y_test, p >= 0.5),
        "test_log_loss": log_loss(y_test, p),
        "weight_norm": norm,
    }
    print(f"{label:30} {row['test_acc']:9.3f} {row['test_log_loss']:9.4f} {norm:12.1f}")
    return row


print(f"\n{'model':30} {'test acc':>9} {'test loss':>9} {'weight norm':>12}")
results = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sk_none = LogisticRegression(C=np.inf, max_iter=10_000).fit(X_train, y_train)
sk_warnings = [str(w.message).splitlines()[0] for w in caught]
results["scikit-learn, C=inf"] = report(
    "scikit-learn, C=inf (none)",
    sk_none.predict_proba(X_test)[:, 1],
    float(np.linalg.norm(sk_none.coef_)),
)
results["gd, 1,000,000 steps"] = report(
    "scratch gd, no penalty, 1e6",
    gd.predict_proba(X_test),
    float(np.linalg.norm(gd.coef_)),
)
reg = LogisticRegressionScratch(l2=1.0 / len(y_train)).fit(X_train, y_train)
results["newton, l2 = 1/n"] = report(
    "scratch newton, l2 = 1/n",
    reg.predict_proba(X_test),
    float(np.linalg.norm(reg.coef_)),
)
print(f"\nwarnings from scikit-learn with C=inf: {sk_warnings or 'none'}")
print(
    f"training accuracy without a penalty: {accuracy_score(y_train, gd.predict(X_train)):.3f}"
)

Path("results").mkdir(exist_ok=True)
Path("results/separation.json").write_text(
    json.dumps(
        {
            "gd_checkpoints": {
                str(k): [gd.loss_history_[k], gd.weight_norm_history_[k]]
                for k in checkpoints
            },
            "models": results,
            "sklearn_warnings": sk_warnings,
        },
        indent=2,
    )
    + "\n",
    newline="\n",
)
steps = sorted(set(range(101)) | set(np.unique(np.logspace(2, 6, 300).astype(int))))
Path("results/separation_curve.json").write_text(
    json.dumps(
        [[int(k), gd.loss_history_[k], gd.weight_norm_history_[k]] for k in steps]
    )
    + "\n",
    newline="\n",
)
