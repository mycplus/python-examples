"""Read data/wine_split.csv and print the results the C++ program must match.

The C++ companion (cpp-examples, machine-learning/k-nearest-neighbors) reads
the same file and prints the same text; both repositories compare against the
same tests/expected/wine_knn.txt.
"""

import csv
import sys
from pathlib import Path

import numpy as np

from data import CSV, standardize
from knn import KNNClassifier, Metric

path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
with path.open(newline="") as f:
    rows = list(csv.reader(f))
names = rows[0][2:]
train = [r for r in rows[1:] if r[0] == "train"]
test = [r for r in rows[1:] if r[0] == "test"]
X_train = np.array([[float(v) for v in r[2:]] for r in train])
X_test = np.array([[float(v) for v in r[2:]] for r in test])
y_train = np.array([int(r[1]) for r in train])
y_test = np.array([int(r[1]) for r in test])
S_train, S_test = standardize(X_train, X_test)

print(
    f"wine: {len(y_train)} training rows, {len(y_test)} test rows, {len(names)} features"
)
for label, Z in (("unscaled", X_train), ("standardized", S_train)):
    share = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=(0, 1))
    print(
        f"share of squared distance from proline, {label}: {share[-1] / share.sum():.1%}"
    )


def correct(A: np.ndarray, B: np.ndarray, k: int, metric: Metric) -> int:
    pred = KNNClassifier(k=k, metric=metric).fit(A, y_train).predict(B)
    return int((pred == y_test).sum())


print("\ncorrect test predictions (of 45)")
print(" k  metric     unscaled  standardized")
for k, metric in (
    (1, "euclidean"),
    (5, "euclidean"),
    (27, "euclidean"),
    (7, "manhattan"),
):
    m: Metric = "manhattan" if metric == "manhattan" else "euclidean"
    print(
        f"{k:2}  {metric:9}  {correct(X_train, X_test, k, m):8}"
        f"  {correct(S_train, S_test, k, m):12}"
    )

dist, idx = KNNClassifier(k=5).fit(S_train, y_train).kneighbors(S_test[:1])
print(f"\nnearest training rows to test row 0 (class {y_test[0]}), standardized")
print("rank  row  class  distance")
for rank, (i, d) in enumerate(zip(idx[0], dist[0], strict=True), start=1):
    print(f"{rank:4}  {i:3}  {y_train[i]:5}  {d:8.6f}")
