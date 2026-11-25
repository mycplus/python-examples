"""The curse of dimensionality, measured two ways.

1. Distance concentration: in random data, the nearest and farthest points
   end up almost the same distance away as the number of dimensions grows.
2. Irrelevant features: appending pure-noise columns to the standardized
   wine data drags KNN's cross-validated accuracy down.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from data import load_wine_split, standardize
from knn import KNNClassifier, pairwise_distances

rng = np.random.default_rng(2026)
print("uniform random points: 1,000 reference points, 100 queries")
print(f"{'dims':>5} {'nearest':>8} {'farthest':>9} {'(far - near) / near':>20}")
concentration = []
for d in (2, 5, 10, 20, 50, 100, 200, 500, 1000):
    points, queries = rng.random((1000, d)), rng.random((100, d))
    dist = pairwise_distances(queries, points)
    near, far = dist.min(axis=1), dist.max(axis=1)
    contrast = float(np.mean((far - near) / near))
    concentration.append([d, contrast])
    print(f"{d:5} {near.mean():8.3f} {far.mean():9.3f} {contrast:20.2f}")

split = load_wine_split()
X_train, y_train = split[0], split[2]
folds = list(
    RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(
        X_train, y_train
    )
)
print("\nwine, standardized, k=27: cross-validated accuracy with noise columns added")
print("(mean over 5 different draws of the noise)")
print(f"{'noise columns':>13} {'accuracy':>9}")
noise_curve = []
for extra in (0, 5, 10, 20, 50, 100, 200, 500):
    total = 0.0
    for _ in range(5):
        X = np.hstack([X_train, rng.standard_normal((len(y_train), extra))])
        for fit_idx, val_idx in folds:
            A, B = standardize(X[fit_idx], X[val_idx])
            pred = KNNClassifier(k=27).fit(A, y_train[fit_idx]).predict(B)
            total += float((pred == y_train[val_idx]).mean())
    accuracy = total / (5 * len(folds))
    noise_curve.append([extra, accuracy])
    print(f"{extra:13} {accuracy:9.3f}")

Path("results").mkdir(exist_ok=True)
Path("results/dimensionality.json").write_text(
    json.dumps({"concentration": concentration, "noise": noise_curve}) + "\n",
    newline="\n",
)
