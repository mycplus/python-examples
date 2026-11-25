"""Logistic regression on the Wisconsin breast cancer data (569 tumours, 30 features).

Baseline, the from-scratch model fitted two ways, and scikit-learn, all with
the same L2 strength: scikit-learn's default C = 1.0 is l2 = 1 / (C * n).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import load_split
from logistic_regression import LogisticRegressionScratch

X_train, X_test, y_train, y_test, names = load_split()
n = len(y_train)
print(
    f"train rows: {n} ({int(y_train.sum())} malignant), "
    f"test rows: {len(y_test)} ({int(y_test.sum())} malignant)\n"
)

C = 1.0
l2 = 1.0 / (C * n)
models = {
    "baseline (class prior)": DummyClassifier(strategy="prior"),
    "scratch, newton": LogisticRegressionScratch(method="newton", l2=l2),
    "scratch, gradient descent": LogisticRegressionScratch(method="gd", l2=l2),
    "scikit-learn (C=1.0)": LogisticRegression(C=C, tol=1e-10, max_iter=10_000),
}
print(
    f"{'model':27} {'acc':>6} {'prec':>6} {'recall':>6} {'F1':>6} {'AUC':>6} {'logloss':>8}"
)
results = {}
proba = {}
for label, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)
    p = p[:, 1] if p.ndim == 2 else p
    pred = (p >= 0.5).astype(int)
    proba[label] = p
    row = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0.0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, p),
        "log_loss": log_loss(y_test, np.clip(p, 1e-15, 1 - 1e-15)),
    }
    results[label] = row
    print(
        f"{label:27} {row['accuracy']:6.3f} {row['precision']:6.3f} {row['recall']:6.3f}"
        f" {row['f1']:6.3f} {row['roc_auc']:6.3f} {row['log_loss']:8.4f}"
    )

newton = models["scratch, newton"]
descent = models["scratch, gradient descent"]
sk = models["scikit-learn (C=1.0)"]
assert isinstance(newton, LogisticRegressionScratch)
assert isinstance(descent, LogisticRegressionScratch)
assert isinstance(sk, LogisticRegression)
print(
    f"\nnewton: {newton.n_iter_} iterations; gradient descent: {descent.n_iter_} iterations"
)
print(
    f"largest weight gap, newton vs scikit-learn:  {np.max(np.abs(newton.coef_ - sk.coef_[0])):.1e}"
)
print(
    f"largest weight gap, gd vs newton:            {np.max(np.abs(descent.coef_ - newton.coef_)):.1e}"
)

cm = confusion_matrix(y_test, (proba["scratch, newton"] >= 0.5).astype(int))
print(
    "\nconfusion matrix, scratch newton at threshold 0.5 (rows: actual, cols: predicted)"
)
print("              benign  malignant")
print(f"benign        {cm[0, 0]:6d}  {cm[0, 1]:9d}")
print(f"malignant     {cm[1, 0]:6d}  {cm[1, 1]:9d}")

print("\nlargest weights (log-odds per standard deviation) and odds ratios:")
for i in np.argsort(-np.abs(newton.coef_))[:5]:
    print(f"  {names[i]:22} {newton.coef_[i]:6.2f}   x{np.exp(newton.coef_[i]):5.2f}")

Path("results").mkdir(exist_ok=True)
Path("results/breast_cancer.json").write_text(
    json.dumps(
        {
            "train_rows": n,
            "test_rows": len(y_test),
            "l2": l2,
            "metrics": results,
            "newton_iterations": newton.n_iter_,
            "gd_iterations": descent.n_iter_,
            "gap_newton_sklearn": float(np.max(np.abs(newton.coef_ - sk.coef_[0]))),
            "gap_gd_newton": float(np.max(np.abs(descent.coef_ - newton.coef_))),
            "confusion_matrix": cm.tolist(),
        },
        indent=2,
    )
    + "\n",
    newline="\n",
)
