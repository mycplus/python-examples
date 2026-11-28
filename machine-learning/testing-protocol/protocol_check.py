"""Run the MYCPLUS testing protocol on one dataset and print what it finds.

Two questions, both answered from real runs:

1. How much does the reported score move when only the train/test split
   changes, and is the gap over a baseline larger than that movement?
2. Does a from-scratch least-squares fit agree with scikit-learn to the
   tolerance the article states?

No network access: the data ships with scikit-learn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLITS = 50
TEST_SIZE = 0.25
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    bunch = load_breast_cancer()
    return bunch.data, bunch.target


def make_model() -> object:
    # Scaling lives inside the pipeline, so it is fitted on training rows only.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs"),
    )


def split_scores(X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    for seed in range(SPLITS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        model = make_model().fit(X_train, y_train)
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        model_scores.append(float(model.score(X_test, y_test)))
        baseline_scores.append(float(baseline.score(X_test, y_test)))
    return {"model": model_scores, "baseline": baseline_scores}


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def tolerance_check(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    """From-scratch least squares against scikit-learn on the same split."""

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # From scratch: add an intercept column and solve in the least-squares sense.
    design = np.column_stack([X_train_scaled, np.ones(len(X_train_scaled))])
    weights, *_ = np.linalg.lstsq(design, y_train.astype(float), rcond=None)
    scratch = design @ weights

    library = LinearRegression().fit(X_train_scaled, y_train.astype(float))
    reference = library.predict(X_train_scaled)

    prediction_scale = max(float(np.abs(reference).max()), 1e-12)
    prediction_gap = float(np.abs(scratch - reference).max())
    weight_gap = float(np.abs(weights[:-1] - library.coef_).max())
    weight_scale = max(float(np.abs(library.coef_).max()), 1e-12)
    return {
        "prediction_gap": prediction_gap,
        "prediction_relative": prediction_gap / prediction_scale,
        "weight_gap": weight_gap,
        "weight_relative": weight_gap / weight_scale,
    }


def split_overlap(y: np.ndarray, seed_a: int = 0, seed_b: int = 1) -> float:
    """How many training rows two of the splits share, on average."""
    index = np.arange(len(y))
    trains = []
    for seed in range(SPLITS):
        train_index, _ = train_test_split(
            index, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        trains.append(set(train_index.tolist()))
    shared = []
    for first in range(SPLITS):
        for second in range(first + 1, SPLITS):
            shared.append(len(trains[first] & trains[second]))
    return float(np.mean(shared))


def main() -> None:
    X, y = load_data()
    n_rows, n_features = X.shape

    print("MYCPLUS testing protocol: split variance and baseline, measured")
    print(f"dataset        breast_cancer: {n_rows} rows, {n_features} features")
    unique, counts = np.unique(y, return_counts=True)
    share = counts / counts.sum()
    print(
        f"class balance  {unique[0]}: {counts[0]} ({share[0]:.1%}), "
        f"{unique[1]}: {counts[1]} ({share[1]:.1%})"
    )
    print(
        f"protocol       {SPLITS} stratified splits, test_size={TEST_SIZE}, "
        f"random_state=0..{SPLITS - 1}"
    )
    print("model          StandardScaler then LogisticRegression (default C=1)")
    print("baseline       DummyClassifier, most frequent class")
    print()

    scores = split_scores(X, y)
    model = describe(scores["model"])
    baseline = describe(scores["baseline"])

    print(f"model accuracy      mean {model['mean']:.4f}  sd {model['sd']:.4f}  "
          f"min {model['min']:.4f}  max {model['max']:.4f}")
    print(f"baseline accuracy   mean {baseline['mean']:.4f}  sd {baseline['sd']:.4f}  "
          f"min {baseline['min']:.4f}  max {baseline['max']:.4f}")
    print(f"spread of one model's score across splits   "
          f"{(model['max'] - model['min']) * 100:.2f} accuracy points")
    perfect = int(sum(score == 1.0 for score in scores["model"]))
    print(f"splits where the model scored 1.0000        {perfect} of {SPLITS}")
    print()

    diffs = np.asarray(scores["model"]) - np.asarray(scores["baseline"])
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))
    standard_error = sd_diff / np.sqrt(SPLITS)
    print(f"paired difference model - baseline   mean {mean_diff:.4f}  "
          f"sd {sd_diff:.4f}")
    print(f"standard error of that difference    {standard_error:.4f}")
    print(f"smallest difference worth reporting  {2 * standard_error:.4f} "
          f"(two standard errors)")
    print(f"splits where the model beats the baseline   "
          f"{(diffs > 0).sum()} of {SPLITS}")
    overlap = split_overlap(y)
    print(f"training rows shared by two splits   {overlap:.1f} of {n_rows} "
          f"({overlap / n_rows:.1%}) on average")
    print()

    best_seed = int(np.argmax(scores["model"]))
    worst_seed = int(np.argmin(scores["model"]))
    print(f"score if only split 0 had been run       {scores['model'][0]:.4f}")
    print(f"highest-scoring split: {best_seed:<3} {scores['model'][best_seed]:.4f}  "
          f"({(scores['model'][best_seed] - model['mean']) * 100:+.2f} points from the mean)")
    print(f"lowest-scoring split:  {worst_seed:<3} {scores['model'][worst_seed]:.4f}  "
          f"({(scores['model'][worst_seed] - model['mean']) * 100:+.2f} points from the mean)")
    print(f"gap between the highest and lowest split     "
          f"{(scores['model'][best_seed] - scores['model'][worst_seed]) * 100:.2f} points")
    print()

    agreement = tolerance_check(X, y)
    print("agreement between the from-scratch fit and scikit-learn, split 0")
    print(f"max prediction gap        {agreement['prediction_gap']:.3e}  "
          f"(relative {agreement['prediction_relative']:.3e})")
    print(f"max coefficient gap       {agreement['weight_gap']:.3e}  "
          f"(relative {agreement['weight_relative']:.3e})")
    print("tolerance used in this series   1e-10 relative for fitted weights")
    print("verdict                         within tolerance"
          if agreement["weight_relative"] < 1e-10
          else "verdict                         outside tolerance")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "breast_cancer",
        "rows": int(n_rows),
        "features": int(n_features),
        "splits": SPLITS,
        "test_size": TEST_SIZE,
        "model": scores["model"],
        "baseline": scores["baseline"],
        "model_summary": model,
        "baseline_summary": baseline,
        "paired_difference": {
            "mean": mean_diff,
            "sd": sd_diff,
            "standard_error": float(standard_error),
            "two_standard_errors": float(2 * standard_error),
        },
        "mean_training_overlap": overlap,
        "agreement": agreement,
    }
    out = RESULTS / "split_scores.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
