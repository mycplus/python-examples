"""A CART decision tree classifier from scratch with NumPy.

fit() grows a binary tree greedily: at every node it tries every feature and
every threshold halfway between two neighboring distinct values, and keeps
the split that lowers the impurity (Gini or entropy) the most.

Ties between equally good splits go to the lower feature index, then to the
lower threshold, so the same data always gives the same tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Criterion = Literal["gini", "entropy"]


def impurity(counts: FloatArray, criterion: Criterion) -> FloatArray:
    """Gini or entropy (in bits) of class counts; counts has classes on the last axis."""
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    if criterion == "gini":
        result: FloatArray = 1.0 - (p * p).sum(axis=-1)
        return result
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    entropy: FloatArray = -terms.sum(axis=-1)
    return entropy


@dataclass
class Split:
    gain: float
    feature: int
    threshold: float


def best_split(
    X: FloatArray, y: IntArray, n_classes: int, criterion: Criterion, min_leaf: int
) -> Split | None:
    """The split of these rows with the largest impurity decrease, or None."""
    n = len(y)
    parent = float(
        impurity(np.bincount(y, minlength=n_classes).astype(float), criterion)
    )
    best: Split | None = None
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable")
        x = X[order, f]
        onehot = np.eye(n_classes)[y[order]]
        left = np.cumsum(onehot, axis=0)[:-1]  # class counts of rows 0..i
        right = left[-1] + onehot[-1] - left
        n_left = np.arange(1, n, dtype=float)
        valid = (x[1:] > x[:-1]) & (n_left >= min_leaf) & (n - n_left >= min_leaf)
        if not valid.any():
            continue
        child = (
            n_left * impurity(left, criterion)
            + (n - n_left) * impurity(right, criterion)
        ) / n
        gain = np.where(valid, parent - child, -np.inf)
        i = int(np.argmax(gain))  # the first maximum: the lowest threshold
        if best is None or gain[i] > best.gain:
            # halfway between the two values, computed as scikit-learn does
            threshold = x[i] / 2.0 + x[i + 1] / 2.0
            best = Split(float(gain[i]), f, float(threshold))
    return best


class DecisionTreeClassifier:
    """Binary classification tree; nodes are stored in parallel arrays.

    For node i: feature_[i] and threshold_[i] give its test (feature_ is -1 at
    a leaf), rows with x[feature] <= threshold go to left_[i], the others to
    right_[i], and counts_[i] holds the training rows of each class there.
    """

    def __init__(
        self,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_split < 2 or min_samples_leaf < 1:
            raise ValueError("min_samples_split must be >= 2 and min_samples_leaf >= 1")
        self.criterion: Criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: ArrayLike, y: ArrayLike) -> DecisionTreeClassifier:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2 or y_arr.shape != (X_arr.shape[0],) or len(y_arr) == 0:
            raise ValueError("X must be 2-D and y must have one label per row")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or infinity")
        self.classes_, codes = np.unique(y_arr, return_inverse=True)
        k = len(self.classes_)
        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        counts: list[FloatArray] = []
        depth_of: list[int] = []

        def grow(rows: IntArray, depth: int) -> int:
            node = len(feature)
            c = np.bincount(codes[rows], minlength=k).astype(float)
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            counts.append(c)
            depth_of.append(depth)
            stop = (
                (self.max_depth is not None and depth >= self.max_depth)
                or len(rows) < self.min_samples_split
                or len(rows) < 2 * self.min_samples_leaf
                or (c > 0).sum() == 1  # pure
            )
            if stop:
                return node
            split = best_split(
                X_arr[rows], codes[rows], k, self.criterion, self.min_samples_leaf
            )
            if split is None:
                return node
            go_left = X_arr[rows, split.feature] <= split.threshold
            if go_left.all() or not go_left.any():  # cannot happen with exact midpoints
                return node
            feature[node], threshold[node] = split.feature, split.threshold
            left[node] = grow(rows[go_left], depth + 1)
            right[node] = grow(rows[~go_left], depth + 1)
            return node

        grow(np.arange(len(y_arr)), 0)
        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.counts_ = np.array(counts)
        self.depth_ = max(depth_of)
        self.n_features_in_ = X_arr.shape[1]
        return self

    @property
    def n_leaves_(self) -> int:
        return int((self.feature_ < 0).sum())

    def apply(self, X: ArrayLike) -> IntArray:
        """The leaf each row lands in."""
        Q = np.asarray(X, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2-D with as many columns as the training data")
        node = np.zeros(len(Q), dtype=np.int64)
        while True:
            inner = self.feature_[node] >= 0
            if not inner.any():
                return node
            rows = np.nonzero(inner)[0]
            n = node[rows]
            go_left = Q[rows, self.feature_[n]] <= self.threshold_[n]
            node[rows] = np.where(go_left, self.left_[n], self.right_[n])

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        c = self.counts_[self.apply(X)]
        proba: FloatArray = c / c.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        # the majority class of the leaf; a tie goes to the smallest label
        result: NDArray[np.generic] = self.classes_[
            np.argmax(self.predict_proba(X), axis=1)
        ]
        return result
