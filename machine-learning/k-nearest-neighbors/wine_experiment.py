"""KNN on the wine data: how much feature scaling matters, and choosing k.

Every choice (scaling, k, distance, vote weights) is made by repeated
cross-validation on the 133 training rows, with the scaler refitted inside
every fold. The 45 test rows are used once, at the end.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from data import FloatArray, load_wine_split, standardize
from knn import KNNClassifier, Metric, Weights

X_train, X_test, y_train, y_test, names = load_wine_split()
KS = list(range(1, 32, 2))
FOLDS = list(
    RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(
        X_train, y_train
    )
)


def cv_curve(scale: bool, metric: Metric, weights: Weights) -> FloatArray:
    """Mean accuracy for every k in KS over 5-fold CV repeated 10 times."""
    scores = np.zeros(len(KS))
    for fit_idx, val_idx in FOLDS:
        A, B = X_train[fit_idx], X_train[val_idx]
        if scale:
            A, B = standardize(A, B)
        for i, k in enumerate(KS):
            model = KNNClassifier(k=k, metric=metric, weights=weights)
            pred = model.fit(A, y_train[fit_idx]).predict(B)
            scores[i] += (pred == y_train[val_idx]).mean()
    result: FloatArray = scores / len(FOLDS)
    return result


# How much of the squared distance between training rows each feature supplies.
for label, Z in (
    ("unscaled", X_train),
    ("standardized", standardize(X_train, X_train)[0]),
):
    share = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=(0, 1))
    share /= share.sum()
    proline = share[names.index("proline")]
    print(f"{label:13} proline's share of squared distance: {proline:.1%}")

variants: dict[str, tuple[bool, Metric, Weights]] = {
    "unscaled, euclidean": (False, "euclidean", "uniform"),
    "standardized, euclidean": (True, "euclidean", "uniform"),
    "standardized, manhattan": (True, "manhattan", "uniform"),
    "standardized, euclidean, distance-weighted": (True, "euclidean", "distance"),
}
curves = {label: cv_curve(*v) for label, v in variants.items()}

print("\ncross-validated accuracy on the training rows (5 folds x 10 repeats)")
print(f"{'k':>3} {'unscaled':>9} {'standardized':>13}")
for i, k in enumerate(KS):
    a, b = curves["unscaled, euclidean"][i], curves["standardized, euclidean"][i]
    print(f"{k:3} {a:9.3f} {b:13.3f}")

print("\nbest k for each variant (the smallest k among equal scores)")
best = {}
for label, curve in curves.items():
    i = int(np.argmax(curve))
    best[label] = KS[i]
    print(f"{label:44} k={KS[i]:<3} {curve[i]:.3f}")

print(f"\ntest rows: {len(y_test)}")
S_train, S_test = standardize(X_train, X_test)
results = {}
for label, (scale, metric, weights) in {
    "scikit-learn defaults: unscaled, k=5": (False, "euclidean", "uniform"),
    **variants,
}.items():
    k = best.get(label, 5)
    A, B = (S_train, S_test) if scale else (X_train, X_test)
    model = KNNClassifier(k=k, metric=metric, weights=weights)
    correct = int((model.fit(A, y_train).predict(B) == y_test).sum())
    results[label] = {"k": k, "correct": correct}
    print(f"{label:44} k={k:<3} {correct}/{len(y_test)} = {correct / len(y_test):.3f}")

# The from-scratch model and scikit-learn must make the same predictions.
metrics: tuple[Metric, ...] = ("euclidean", "manhattan")
weightings: tuple[Weights, ...] = ("uniform", "distance")
settings = [(k, m, w) for k in KS for m in metrics for w in weightings]
same = 0
for k, m, w in settings:
    ours = KNNClassifier(k=k, metric=m, weights=w).fit(S_train, y_train)
    sk = KNeighborsClassifier(n_neighbors=k, metric=m, weights=w, algorithm="brute")
    sk.fit(S_train, y_train)
    same += bool((ours.predict(S_test) == sk.predict(S_test)).all())
print(f"\nsame test predictions as scikit-learn: {same} of {len(settings)} settings")

Path("results").mkdir(exist_ok=True)
Path("results/wine_cv_curve.json").write_text(
    json.dumps(
        {
            "k": KS,
            "curves": {label: c.tolist() for label, c in curves.items()},
            "best": best,
            "test": results,
        }
    )
    + "\n",
    newline="\n",
)
