"""Time brute-force KNN prediction four ways on the same synthetic data.

20,000 training rows, 8 features, 3 classes, k = 5, one thread everywhere.
The C++ program in cpp-examples generates the same data with the same
splitmix64 stream and prints the same checksum, so all versions are checked
to make the same 1,000 predictions before any time is compared.
"""

import math
import platform
import statistics
import time
from collections.abc import Callable

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from data import synthetic
from knn import KNNClassifier

N_TRAIN, N_QUERY, DIMS, K = 20_000, 1_000, 8, 5
X, y = synthetic(N_TRAIN, DIMS, seed=1)
Q, q_labels = synthetic(N_QUERY, DIMS, seed=2)


def checksum(pred: np.ndarray) -> int:
    return int(sum((i + 1) * int(p) for i, p in enumerate(pred)))


def pure_python(queries: list[list[float]]) -> list[int]:
    """No NumPy: lists, math.dist and a sort."""
    rows, labels = X.tolist(), y.tolist()
    out = []
    for q in queries:
        order = sorted(range(len(rows)), key=lambda i: (math.dist(q, rows[i]), i))
        votes = [0, 0, 0]
        for i in order[:K]:
            votes[labels[i]] += 1
        out.append(votes.index(max(votes)))
    return out


def median_seconds(run: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


with threadpool_limits(limits=1):
    ours = KNNClassifier(k=K).fit(X, y)
    brute = KNeighborsClassifier(K, algorithm="brute").fit(X, y)
    tree = KNeighborsClassifier(K, algorithm="kd_tree").fit(X, y)
    pred = ours.predict(Q)
    print(f"{N_TRAIN:,} training rows, {N_QUERY:,} queries, {DIMS} features, k={K}")
    print(
        f"checksum of predictions: {checksum(pred)}; accuracy {np.mean(pred == q_labels):.3f}"
    )
    agree = [
        (brute.predict(Q) == pred).all(),
        (tree.predict(Q) == pred).all(),
        pure_python(Q[:50].tolist()) == pred[:50].tolist(),
    ]
    print(f"scikit-learn brute, k-d tree and pure Python agree: {all(agree)}")

    sample = Q[:20].tolist()
    timings = {
        "pure Python (20 queries)": median_seconds(lambda: pure_python(sample), 3) / 20,
        "NumPy, this article's KNNClassifier": median_seconds(
            lambda: ours.predict(Q), 5
        )
        / N_QUERY,
        "scikit-learn, algorithm='brute'": median_seconds(lambda: brute.predict(Q), 5)
        / N_QUERY,
        "scikit-learn, algorithm='kd_tree'": median_seconds(lambda: tree.predict(Q), 5)
        / N_QUERY,
    }

print(f"\nmicroseconds per query (median of repeated runs), {platform.machine()}")
for label, seconds in timings.items():
    print(f"{label:38} {seconds * 1e6:10.1f}")
