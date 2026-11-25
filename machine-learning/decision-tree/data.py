"""The breast cancer split shared by every experiment (the same split as the
logistic regression article), the files the C++ companion reads, and the
row order the Python and C++ benchmarks share."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
HERE = Path(__file__).resolve().parent


def load_split() -> tuple[FloatArray, FloatArray, IntArray, IntArray, list[str]]:
    """80/20 stratified split: 455 training rows, 114 test rows.

    scikit-learn codes malignant as 0; the labels are flipped so that
    1 = malignant, the class a screening must not miss.
    """
    data = load_breast_cancer()
    y = (1 - data.target).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def write_test_csv(path: Path = HERE / "data" / "breast_cancer_test.csv") -> None:
    """The 114 test rows as CSV for the C++ program: class, then 30 features."""
    _, X_test, _, y_test, names = load_split()
    lines = ["class," + ",".join(n.replace(" ", "_") for n in names)]
    for row, label in zip(X_test, y_test, strict=True):
        lines.append(f"{label}," + ",".join(repr(float(v)) for v in row))
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", newline="\n")


class SplitMix64:
    """splitmix64: the same stream of numbers in Python and in the C++ benchmark."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)


def benchmark_order(n_rows: int = 1_000_008, n_test: int = 114) -> IntArray:
    """Which test row to use for each benchmark row, in a pseudo-random order,
    so that the processor cannot learn a repeating pattern of branches."""
    rng = SplitMix64(2026)
    return np.array([rng.next() % n_test for _ in range(n_rows)], dtype=np.int64)
