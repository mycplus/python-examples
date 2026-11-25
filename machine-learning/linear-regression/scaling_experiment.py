"""How feature scaling changes gradient descent on the diabetes data.

Both runs use the same 353 training rows, the same model and the safe step
1/L. Only the units of the features differ.
"""

import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from linear_regression import (
    LinearRegressionScratch,
    add_intercept,
    lipschitz_constant,
    mse,
)

X, y = load_diabetes(scaled=False, return_X_y=True)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
inputs = {
    "raw units": X_train,
    "standardized": StandardScaler().fit(X_train).transform(X_train),
}
MAX_ITER = 1_000_000

print(
    f"{'features':13} {'condition no.':>14} {'step 1/L':>11} {'iterations':>11}"
    f" {'converged':>10} {'loss gap':>9} {'seconds':>8}"
)
summary, histories = {}, {}
for label, features in inputs.items():
    Xb = add_intercept(features)
    hessian_eigs = np.linalg.eigvalsh((2.0 / len(y_train)) * (Xb.T @ Xb))
    condition = hessian_eigs[-1] / hessian_eigs[0]
    best = mse(Xb, y_train, np.linalg.lstsq(Xb, y_train)[0])

    model = LinearRegressionScratch(method="gd", max_iter=MAX_ITER)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(features, y_train)
    seconds = time.perf_counter() - start
    gap = (model.loss_history_[-1] - best) / best
    print(
        f"{label:13} {condition:14.3g} {1 / lipschitz_constant(Xb):11.3g}"
        f" {model.n_iter_:11,} {model.converged_!s:>10} {gap:9.1e}"
        f" {seconds:8.1f}"
    )
    summary[label] = {
        "condition_number": float(condition),
        "step": float(model.learning_rate_),
        "iterations": model.n_iter_,
        "converged": model.converged_,
        "final_relative_loss_gap": float(gap),
        "seconds": seconds,
    }
    histories[label] = (np.array(model.loss_history_) - best) / best

# Relative loss gap after a fixed number of steps, for the article's table.
checkpoints = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
print("\nrelative loss gap (loss - best) / best after k steps:")
print(f"{'k':>9} {'raw units':>11} {'standardized':>13}")
for k in checkpoints:
    cells = []
    for label in inputs:
        h = histories[label]
        cells.append(f"{h[k]:.1e}" if k < len(h) else "converged")
    print(f"{k:9,} {cells[0]:>11} {cells[1]:>13}")

Path("results").mkdir(exist_ok=True)
Path("results/scaling.json").write_text(json.dumps(summary, indent=2) + "\n")
# Every step up to 100, then log-spaced, for the convergence figure.
steps = sorted(set(range(101)) | set(np.unique(np.logspace(2, 6, 400).astype(int))))
with open("results/gd_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "raw_gap", "standardized_gap"])
    for k in steps:
        row = [k]
        for label in inputs:
            h = histories[label]
            row.append(f"{h[k]:.6e}" if k < len(h) else "")
        writer.writerow(row)
