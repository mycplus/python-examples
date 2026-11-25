"""Time prediction with the exported tree (entropy, depth 3) three ways.

1,000,008 rows are drawn from the 114 test rows in a pseudo-random order that
the C++ benchmark in cpp-examples reproduces. Each version must produce the
same checksum, which the C++ benchmark also prints, before any time is
compared. One thread everywhere.
"""

import platform
import statistics
import time
from collections.abc import Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from threadpoolctl import threadpool_limits

from data import HERE, benchmark_order, load_split
from predict_exported import load_tree, predict
from tree import DecisionTreeClassifier

split = load_split()
X_train, X_test, y_train = split[0], split[1], split[2]
rows = X_test[benchmark_order()]
nodes = load_tree(HERE / "models" / "tree.txt")
ours = DecisionTreeClassifier("entropy", max_depth=3).fit(X_train, y_train)
sk = SklearnTree(criterion="entropy", max_depth=3, random_state=0).fit(X_train, y_train)


def checksum(pred: np.ndarray | list[int]) -> int:
    return int(sum((i + 1) * int(p) for i, p in enumerate(pred)) % 1_000_000_007)


def median_seconds(run: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


as_lists = rows.tolist()
with threadpool_limits(limits=1):
    results: dict[str, Callable[[], object]] = {
        "pure Python, exported node table": lambda: [
            predict(nodes, r)[0] for r in as_lists
        ],
        "NumPy, this article's tree": lambda: ours.predict(rows),
        "scikit-learn predict": lambda: sk.predict(rows),
    }
    sums = {label: checksum(np.asarray(run())) for label, run in results.items()}
    print(f"{len(rows):,} rows, tree depth 3, 7 leaves")
    print(
        f"checksums: {sorted(set(sums.values()))} (all equal: {len(set(sums.values())) == 1})"
    )
    print(f"\nnanoseconds per row (median of repeated runs), {platform.machine()}")
    for label, run in results.items():
        repeats = 3 if label.startswith("pure") else 7
        seconds = median_seconds(run, repeats)
        print(f"{label:36} {seconds / len(rows) * 1e9:9.1f}")
