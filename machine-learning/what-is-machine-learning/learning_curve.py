"""Does more training data help? Measure it instead of assuming it.

For each training fraction, fit the same pipeline on that many training rows and
score it on the held-out rows, repeated over 20 split seeds. Writes
results/learning_curve.json for the figure and prints a table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REPEATS = 20
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    iris = load_iris()
    rows = []
    for fraction in FRACTIONS:
        scores: list[float] = []
        for seed in range(REPEATS):
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target, test_size=0.25, random_state=seed, stratify=iris.target
            )
            keep = max(round(len(X_train) * fraction), 12)
            subset = np.random.default_rng(seed).choice(len(X_train), keep, replace=False)
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000)
            ).fit(X_train[subset], y_train[subset])
            scores.append(float(accuracy_score(y_test, model.predict(X_test))))
        values = np.asarray(scores)
        rows.append(
            {
                "fraction": fraction,
                "training_rows": keep,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    baseline_scores = []
    for seed in range(REPEATS):
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.25, random_state=seed, stratify=iris.target
        )
        dumb = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline_scores.append(float(accuracy_score(y_test, dumb.predict(X_test))))

    print("learning curve: same pipeline, more training rows, 20 split seeds per row")
    print(f"baseline accuracy over those {REPEATS} splits   "
          f"mean {np.mean(baseline_scores):.4f}  sd {np.std(baseline_scores, ddof=1):.4f}")
    print()
    print("training rows   fraction   mean accuracy   sd       min      max")
    for row in rows:
        print(f"{row['training_rows']:>13}   {row['fraction']:>8.1f}   {row['mean']:>13.4f}   "
              f"{row['sd']:.4f}   {row['min']:.4f}   {row['max']:.4f}")

    tail = [row for row in rows if row["fraction"] >= 0.5]
    plateau = sum(row["mean"] for row in tail) / len(tail)
    print()
    print(f"mean accuracy over the six largest fractions   {plateau:.4f}")
    print(f"rows where the curve flattens                  "
          f"{min(row['training_rows'] for row in tail)}")
    print(f"gain from the smallest fraction to the largest "
          f"{rows[-1]['mean'] - rows[0]['mean']:.4f}")
    print(f"standard deviation, smallest to largest        "
          f"{rows[0]['sd']:.4f} to {rows[-1]['sd']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "iris",
        "repeats": REPEATS,
        "fractions": rows,
        "baseline_mean": float(np.mean(baseline_scores)),
        "baseline_sd": float(np.std(baseline_scores, ddof=1)),
        "plateau_mean": float(plateau),
        "plateau_from_rows": int(min(row["training_rows"] for row in tail)),
    }
    out = RESULTS / "learning_curve.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
