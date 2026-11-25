"""The breast cancer split shared by every experiment."""

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]


def load_raw_split() -> tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, list[str]
]:
    """Unscaled 80/20 stratified split; label 1 = malignant, 0 = benign.

    scikit-learn codes malignant as 0, so the labels are flipped: the class a
    screening test must not miss becomes the positive class.
    """
    data = load_breast_cancer()
    X, y = data.data, 1.0 - data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def load_split() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, list[str]]:
    """The same split, standardized with a scaler fitted on the training rows only."""
    X_train, X_test, y_train, y_test, names = load_raw_split()
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_test),
        y_train,
        y_test,
        names,
    )
