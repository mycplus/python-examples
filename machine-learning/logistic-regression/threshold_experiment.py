"""Choosing the decision threshold without touching the test set.

Out-of-fold probabilities from 5-fold cross-validation on the training rows
show how precision and recall trade off. The threshold is picked there (the
highest one that keeps recall >= 0.98) and only then applied to the test rows.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data import load_raw_split
from logistic_regression import LogisticRegressionScratch

X_train, X_test, y_train, y_test, _ = load_raw_split()
l2 = 1.0 / len(y_train)  # the same strength as scikit-learn's default C = 1.0

oof = np.empty(len(y_train))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fit_idx, val_idx in folds.split(X_train, y_train):
    scaler = StandardScaler().fit(X_train[fit_idx])  # per fold: no leakage
    model = LogisticRegressionScratch(l2=1.0 / len(fit_idx))
    model.fit(scaler.transform(X_train[fit_idx]), y_train[fit_idx])
    oof[val_idx] = model.predict_proba(scaler.transform(X_train[val_idx]))

print("out-of-fold results on the 455 training rows (170 malignant):")
print(
    f"{'threshold':>9} {'precision':>9} {'recall':>7} {'missed':>7} {'false alarms':>13}"
)
rows = []
for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    pred = (oof >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
    prec, rec = precision_score(y_train, pred), recall_score(y_train, pred)
    rows.append(
        {"threshold": t, "precision": prec, "recall": rec, "fn": int(fn), "fp": int(fp)}
    )
    print(f"{t:9.1f} {prec:9.3f} {rec:7.3f} {fn:7d} {fp:13d}")

candidates = np.round(np.arange(0.01, 1.0, 0.01), 2)
ok = [t for t in candidates if recall_score(y_train, (oof >= t).astype(int)) >= 0.98]
chosen = float(max(ok))
print(f"\nhighest threshold with out-of-fold recall >= 0.98: {chosen:.2f}")

scaler = StandardScaler().fit(X_train)
final = LogisticRegressionScratch(l2=l2).fit(scaler.transform(X_train), y_train)
p_test = final.predict_proba(scaler.transform(X_test))
print(f"\ntest rows: {len(y_test)} ({int(y_test.sum())} malignant)")
print(
    f"{'threshold':>9} {'precision':>9} {'recall':>7} {'missed':>7} {'false alarms':>13}"
)
test_rows = []
for t in (0.5, chosen):
    pred = (p_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    prec, rec = precision_score(y_test, pred), recall_score(y_test, pred)
    test_rows.append(
        {"threshold": t, "precision": prec, "recall": rec, "fn": int(fn), "fp": int(fp)}
    )
    print(f"{t:9.2f} {prec:9.3f} {rec:7.3f} {fn:7d} {fp:13d}")

Path("results").mkdir(exist_ok=True)
Path("results/threshold.json").write_text(
    json.dumps({"oof": rows, "chosen": chosen, "test": test_rows}, indent=2) + "\n",
    newline="\n",
)
# Precision and recall at every 0.01 for the figure.
curve = []
for t in candidates:
    pred = (oof >= t).astype(int)
    curve.append(
        [
            float(t),
            precision_score(y_train, pred, zero_division=1.0),
            recall_score(y_train, pred),
        ]
    )
Path("results/threshold_curve.json").write_text(json.dumps(curve) + "\n", newline="\n")
