"""The wine split shared by every experiment, and a synthetic data generator
that the C++ companion reproduces bit for bit."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CSV = Path(__file__).resolve().parent / "data" / "wine_split.csv"


def load_wine_split() -> tuple[FloatArray, FloatArray, IntArray, IntArray, list[str]]:
    """Unscaled 75/25 stratified split: 133 training rows, 45 test rows."""
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def standardize(
    X_train: FloatArray, X_test: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Scale every feature to mean 0 and standard deviation 1 on the training rows."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)  # population standard deviation, as StandardScaler
    return (X_train - mean) / std, (X_test - mean) / std


def write_csv(path: Path = CSV) -> None:
    """The split as CSV for the C++ program: split, class, then 13 raw features."""
    X_train, X_test, y_train, y_test, names = load_wine_split()
    lines = ["split,class," + ",".join(n.replace("/", "_") for n in names)]
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        for row, label in zip(X, y, strict=True):
            lines.append(f"{split},{label}," + ",".join(repr(float(v)) for v in row))
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", newline="\n")


class SplitMix64:
    """splitmix64: the same stream of numbers in Python and in the C++ program."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def uniform(self) -> float:
        """A double in [0, 1) from the top 53 bits."""
        return (self.next() >> 11) * 2.0**-53


def synthetic(n_rows: int, n_features: int, seed: int) -> tuple[FloatArray, IntArray]:
    """Uniform points in the unit cube; the class is which third of the cube
    the sum of the first two coordinates falls in (0, 1 or 2)."""
    rng = SplitMix64(seed)
    X = np.array([[rng.uniform() for _ in range(n_features)] for _ in range(n_rows)])
    s = X[:, 0] + X[:, 1]
    y = (s >= 2.0 / 3.0).astype(np.int64) + (s >= 4.0 / 3.0).astype(np.int64)
    return X, y
