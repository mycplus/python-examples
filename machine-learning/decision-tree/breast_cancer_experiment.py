"""Decision trees on the breast cancer data: depth, Gini vs entropy, scaling.

Depth and criterion are chosen by 5-fold cross-validation repeated 10 times on
the 455 training rows; the 114 test rows are used once, at the end.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from data import load_split
from export import export_text
from tree import Criterion, DecisionTreeClassifier

X_train, X_test, y_train, y_test, names = load_split()
DEPTHS: list[int | None] = [1, 2, 3, 4, 5, 6, 7, 8, None]
FOLDS = list(
    RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(
        X_train, y_train
    )
)


def cv_scores(criterion: Criterion, depth: int | None) -> np.ndarray:
    """Accuracy on each of the 50 validation folds."""
    scores = []
    for fit_idx, val_idx in FOLDS:
        tree = DecisionTreeClassifier(criterion, depth).fit(
            X_train[fit_idx], y_train[fit_idx]
        )
        scores.append((tree.predict(X_train[val_idx]) == y_train[val_idx]).mean())
    return np.array(scores)


print(
    f"training rows: {len(y_train)}, test rows: {len(y_test)} ({int(y_test.sum())} malignant)"
)
print(
    "\ncross-validated accuracy (5 folds x 10 repeats) and size of the tree on all training rows"
)
print(
    f"{'depth':>5} {'gini CV':>8} {'leaves':>6} {'train':>6}   {'entropy CV':>10} {'leaves':>6} {'train':>6}"
)
table: dict[str, list[dict[str, float]]] = {"gini": [], "entropy": []}
for depth in DEPTHS:
    cells = []
    for criterion in ("gini", "entropy"):
        scores = cv_scores(criterion, depth)
        full = DecisionTreeClassifier(criterion, depth).fit(X_train, y_train)
        train_acc = float((full.predict(X_train) == y_train).mean())
        table[criterion].append(
            {
                "depth": depth or 0,
                "cv": float(scores.mean()),
                "se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                "leaves": full.n_leaves_,
                "train": train_acc,
            }
        )
        cells.append(f"{scores.mean():8.3f} {full.n_leaves_:6} {train_acc:6.3f}")
    label = "none" if depth is None else str(depth)
    print(f"{label:>5} {cells[0]}   {'':2}{cells[1]}")

# Pick the smallest tree within one standard error of the best score.
print("\nsmallest depth within one standard error of the best CV score")
chosen: dict[str, int] = {}
for criterion, rows in table.items():
    best = max(rows, key=lambda r: r["cv"])
    pick = next(r for r in rows if r["cv"] >= best["cv"] - best["se"])
    chosen[criterion] = int(pick["depth"])
    print(
        f"{criterion:8} best {best['cv']:.3f} +/- {best['se']:.3f} at depth "
        f"{int(best['depth']) or 'none'}; chosen depth {int(pick['depth'])} ({pick['cv']:.3f})"
    )

print(f"\ntest rows: {len(y_test)}")
models = {
    "baseline (always benign)": DummyClassifier(strategy="most_frequent"),
    "gini, no depth limit": DecisionTreeClassifier("gini"),
    f"gini, depth {chosen['gini']}": DecisionTreeClassifier("gini", chosen["gini"]),
    f"entropy, depth {chosen['entropy']}": DecisionTreeClassifier(
        "entropy", chosen["entropy"]
    ),
}
predictions = {}
for label, model in models.items():
    pred = model.fit(X_train, y_train).predict(X_test)
    predictions[label] = pred
    missed = int(((pred == 0) & (y_test == 1)).sum())
    false_alarms = int(((pred == 1) & (y_test == 0)).sum())
    print(
        f"{label:28} {int((pred == y_test).sum()):3}/{len(y_test)} correct,"
        f" {missed} malignant missed, {false_alarms} false alarms"
    )
a, b = list(predictions.values())[2:4]
print(
    f"gini and entropy trees agree on {int((a == b).sum())} of {len(y_test)} test rows"
)

# Trees compare one feature with a threshold, so rescaling a feature moves
# its thresholds but cannot change which rows go left or right.
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
raw = DecisionTreeClassifier("gini").fit(X_train, y_train)
scaled = DecisionTreeClassifier("gini").fit((X_train - mean) / std, y_train)
same = bool((raw.predict(X_test) == scaled.predict((X_test - mean) / std)).all())
print(
    f"\nstandardized features: same features at every node "
    f"{bool((raw.feature_ == scaled.feature_).all())}, same test predictions {same}"
)

depth = chosen["entropy"]
print(f"\nthe entropy tree of depth {depth} (class counts: benign, malignant)")
print(export_text(models[f"entropy, depth {depth}"], names), end="")

Path("results").mkdir(exist_ok=True)
Path("results/depth_curve.json").write_text(
    json.dumps({"table": table, "chosen": chosen}) + "\n", newline="\n"
)
