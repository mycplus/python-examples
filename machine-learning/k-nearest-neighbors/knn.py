"""k-nearest neighbors classification from scratch with NumPy.

fit() only stores the training rows. predict() measures the distance from each
query to every training row, takes the k closest and lets them vote.

Ordering rules, chosen so that results never depend on chance:
  neighbors with equal distances are ordered by training-row index;
  a tied vote goes to the smallest class label (as in scikit-learn).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Metric = Literal["euclidean", "manhattan"]
Weights = Literal["uniform", "distance"]


def pairwise_distances(
    A: FloatArray, B: FloatArray, metric: Metric = "euclidean"
) -> FloatArray:
    """Distance from every row of A to every row of B, shape (len(A), len(B))."""
    out = np.empty((A.shape[0], B.shape[0]))
    for start in range(0, A.shape[0], 256):  # 256 query rows at a time
        diff = A[start : start + 256, None, :] - B[None, :, :]
        if metric == "euclidean":
            out[start : start + 256] = np.sqrt(np.einsum("qnd,qnd->qn", diff, diff))
        else:
            out[start : start + 256] = np.abs(diff).sum(axis=2)
    return out


def nearest(distances: FloatArray, k: int) -> IntArray:
    """Indices of the k smallest distances in each row; ties go to the lower index."""
    order = np.argsort(distances, axis=1, kind="stable")
    return order[:, :k].astype(np.int64)


def vote(
    labels: IntArray, distances: FloatArray, n_classes: int, weights: Weights
) -> FloatArray:
    """Each row's share of the vote per class, from its neighbors' labels."""
    if weights == "uniform":
        w = np.ones_like(distances)
    else:
        with np.errstate(divide="ignore"):
            w = 1.0 / distances
        exact = distances == 0.0  # a neighbor at distance 0 takes the whole vote
        has_exact = exact.any(axis=1)
        w[has_exact] = exact[has_exact].astype(np.float64)
    shares = np.zeros((labels.shape[0], n_classes))
    np.add.at(shares, (np.arange(labels.shape[0])[:, None], labels), w)
    total: FloatArray = shares / shares.sum(axis=1, keepdims=True)
    return total


class KNNClassifier:
    """Majority vote of the k training rows closest to each query."""

    def __init__(
        self, k: int = 5, metric: Metric = "euclidean", weights: Weights = "uniform"
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.metric: Metric = metric
        self.weights: Weights = weights

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNClassifier:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2 or y_arr.shape != (X_arr.shape[0],):
            raise ValueError("X must be 2-D and y must have one label per row")
        if self.k > X_arr.shape[0]:
            raise ValueError("k is larger than the number of training rows")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or infinity")
        self.classes_, self.y_codes_ = np.unique(y_arr, return_inverse=True)
        self.X_ = X_arr
        return self

    def kneighbors(self, X: ArrayLike) -> tuple[FloatArray, IntArray]:
        """Distances to and indices of the k nearest training rows, nearest first."""
        Q = np.asarray(X, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != self.X_.shape[1]:
            raise ValueError("X must be 2-D with as many columns as the training data")
        d = pairwise_distances(Q, self.X_, self.metric)
        idx = nearest(d, self.k)
        return np.take_along_axis(d, idx, axis=1), idx

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        dist, idx = self.kneighbors(X)
        labels = self.y_codes_[idx].astype(np.int64)
        return vote(labels, dist, len(self.classes_), self.weights)

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        # argmax returns the first maximum: a tied vote goes to the smallest label
        winners = np.argmax(self.predict_proba(X), axis=1)
        result: NDArray[np.generic] = self.classes_[winners]
        return result
