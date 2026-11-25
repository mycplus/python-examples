"""Linear regression from scratch with NumPy.

One model, two ways to fit it:
  "lstsq" solves the least-squares problem directly (SVD-based, stable).
  "gd"    runs batch gradient descent on the mean squared error.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def add_intercept(X: FloatArray) -> FloatArray:
    """Prepend a column of ones so that theta[0] is the intercept."""
    return np.column_stack([np.ones(X.shape[0]), X])


def mse(Xb: FloatArray, y: FloatArray, theta: FloatArray) -> float:
    """Mean squared error of the predictions Xb @ theta."""
    residuals = Xb @ theta - y
    return float(residuals @ residuals) / len(y)


def gradient(Xb: FloatArray, y: FloatArray, theta: FloatArray) -> FloatArray:
    """Gradient of the mean squared error with respect to theta."""
    return (2.0 / len(y)) * (Xb.T @ (Xb @ theta - y))


def lipschitz_constant(Xb: FloatArray) -> float:
    """Largest eigenvalue of the MSE Hessian (2/n) Xb^T Xb.

    Gradient descent with a fixed step converges for any step below 2/L;
    1/L is the safe default.
    """
    hessian = (2.0 / Xb.shape[0]) * (Xb.T @ Xb)
    return float(np.linalg.eigvalsh(hessian)[-1])


def _validate(X: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(
            "X must be 2-D (n_samples, n_features); "
            "use X.reshape(-1, 1) for a single feature"
        )
    if y_arr.ndim != 1:
        raise ValueError("y must be 1-D (n_samples,)")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} values"
        )
    if X_arr.shape[0] == 0:
        raise ValueError("cannot fit a model to zero samples")
    if not (np.isfinite(X_arr).all() and np.isfinite(y_arr).all()):
        raise ValueError("X and y must not contain NaN or infinity")
    return X_arr, y_arr


class LinearRegressionScratch:
    """Ordinary least squares: y_hat = intercept_ + X @ coef_."""

    def __init__(
        self,
        method: str = "lstsq",
        learning_rate: float | None = None,
        max_iter: int = 1_000_000,
        tol: float = 1e-9,
    ) -> None:
        if method not in ("lstsq", "gd"):
            raise ValueError("method must be 'lstsq' or 'gd'")
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegressionScratch:
        X_arr, y_arr = _validate(X, y)
        Xb = add_intercept(X_arr)
        if self.method == "lstsq":
            theta = np.linalg.lstsq(Xb, y_arr)[0]
            self.n_iter_ = 0
            self.converged_ = True
            self.loss_history_ = [mse(Xb, y_arr, theta)]
        else:
            theta = self._gradient_descent(Xb, y_arr)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def _gradient_descent(self, Xb: FloatArray, y: FloatArray) -> FloatArray:
        lr = self.learning_rate
        if lr is None:
            lr = 1.0 / lipschitz_constant(Xb)
        theta = np.zeros(Xb.shape[1])
        grad = gradient(Xb, y, theta)
        stop_at = self.tol * float(np.linalg.norm(grad))
        history = [mse(Xb, y, theta)]
        converged = float(np.linalg.norm(grad)) <= stop_at
        iteration = 0
        with np.errstate(over="ignore", invalid="ignore"):
            while not converged and iteration < self.max_iter:
                iteration += 1
                theta = theta - lr * grad
                grad = gradient(Xb, y, theta)
                loss = mse(Xb, y, theta)
                if not np.isfinite(loss):
                    raise FloatingPointError(
                        f"gradient descent diverged at iteration {iteration}; "
                        "lower the learning rate"
                    )
                history.append(loss)
                converged = float(np.linalg.norm(grad)) <= stop_at
        if not converged:
            warnings.warn(
                f"gradient descent stopped after {self.max_iter} iterations "
                "without converging; scale the features or raise max_iter",
                RuntimeWarning,
                stacklevel=3,
            )
        self.learning_rate_ = lr
        self.n_iter_ = iteration
        self.converged_ = converged
        self.loss_history_ = history
        return theta

    def predict(self, X: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X must have shape (n_samples, {len(self.coef_)})")
        return X_arr @ self.coef_ + self.intercept_
