"""Binary logistic regression from scratch with NumPy.

One model, two ways to fit it:
  "newton" takes Newton steps on the log loss (fast, needs a p x p solve).
  "gd"     runs batch gradient descent on the same loss.
An optional L2 penalty (l2 > 0) keeps the weights finite on separable data.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def sigmoid(z: FloatArray) -> FloatArray:
    """1 / (1 + e^-z), computed without overflow for large |z|."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def add_intercept(X: FloatArray) -> FloatArray:
    """Prepend a column of ones so that theta[0] is the intercept."""
    return np.column_stack([np.ones(X.shape[0]), X])


def log_loss(
    Xb: FloatArray, y: FloatArray, theta: FloatArray, l2: float = 0.0
) -> float:
    """Mean binary cross-entropy, plus (l2 / 2) * ||w||^2 (intercept not penalized)."""
    z = Xb @ theta
    # log(1 + e^z) - y * z equals -[y log p + (1 - y) log(1 - p)], without log(0)
    data = float(np.mean(np.logaddexp(0.0, z) - y * z))
    return data + 0.5 * l2 * float(theta[1:] @ theta[1:])


def gradient(
    Xb: FloatArray, y: FloatArray, theta: FloatArray, l2: float = 0.0
) -> FloatArray:
    """Gradient of log_loss with respect to theta."""
    g = Xb.T @ (sigmoid(Xb @ theta) - y) / len(y)
    g[1:] += l2 * theta[1:]
    return g


def hessian(Xb: FloatArray, theta: FloatArray, l2: float = 0.0) -> FloatArray:
    """Second derivatives of log_loss: Xb^T S Xb / n + l2 (S = p (1 - p) on the diagonal)."""
    p = sigmoid(Xb @ theta)
    h: FloatArray = (Xb * (p * (1.0 - p))[:, None]).T @ Xb / float(Xb.shape[0])
    h[1:, 1:] += l2 * np.eye(Xb.shape[1] - 1)
    return h


def lipschitz_constant(Xb: FloatArray, l2: float = 0.0) -> float:
    """Upper bound on the curvature: p (1 - p) <= 1/4, so L = lambda_max(Xb^T Xb) / (4n) + l2."""
    largest = float(np.linalg.eigvalsh(Xb.T @ Xb)[-1])
    return largest / (4.0 * float(Xb.shape[0])) + l2


def _validate(X: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(
            "X must be 2-D (n_samples, n_features); "
            "use X.reshape(-1, 1) for a single feature"
        )
    if y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"y must be 1-D with {X_arr.shape[0]} labels")
    if X_arr.shape[0] == 0:
        raise ValueError("cannot fit a model to zero samples")
    if not np.isfinite(X_arr).all():
        raise ValueError("X must not contain NaN or infinity")
    if not np.isin(y_arr, (0.0, 1.0)).all():
        raise ValueError("y must contain only the labels 0 and 1")
    return X_arr, y_arr


class LogisticRegressionScratch:
    """P(y = 1 | x) = sigmoid(intercept_ + x @ coef_)."""

    def __init__(
        self,
        method: str = "newton",
        l2: float = 0.0,
        learning_rate: float | None = None,
        max_iter: int | None = None,
        tol: float = 1e-8,
    ) -> None:
        if method not in ("newton", "gd"):
            raise ValueError("method must be 'newton' or 'gd'")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.method = method
        self.l2 = l2
        self.learning_rate = learning_rate
        self.max_iter = (
            max_iter
            if max_iter is not None
            else (100 if method == "newton" else 1_000_000)
        )
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegressionScratch:
        X_arr, y_arr = _validate(X, y)
        Xb = add_intercept(X_arr)
        theta = np.zeros(Xb.shape[1])
        grad = gradient(Xb, y_arr, theta, self.l2)
        stop_at = self.tol * float(np.linalg.norm(grad))
        lr = self.learning_rate
        if self.method == "gd" and lr is None:
            lr = 1.0 / lipschitz_constant(Xb, self.l2)
        history = [log_loss(Xb, y_arr, theta, self.l2)]
        norms = [0.0]
        converged = float(np.linalg.norm(grad)) <= stop_at
        iteration = 0
        while not converged and iteration < self.max_iter:
            iteration += 1
            if self.method == "newton":
                step = np.linalg.lstsq(hessian(Xb, theta, self.l2), grad)[0]
                theta = theta - step
            else:
                assert lr is not None
                theta = theta - lr * grad
            grad = gradient(Xb, y_arr, theta, self.l2)
            history.append(log_loss(Xb, y_arr, theta, self.l2))
            norms.append(float(np.linalg.norm(theta[1:])))
            converged = float(np.linalg.norm(grad)) <= stop_at
        if not converged:
            warnings.warn(
                f"{self.method} stopped after {self.max_iter} iterations without "
                "converging; the classes may be separable (try l2 > 0)",
                RuntimeWarning,
                stacklevel=2,
            )
        self.learning_rate_ = lr
        self.n_iter_ = iteration
        self.converged_ = converged
        self.loss_history_ = history
        self.weight_norm_history_ = norms
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Log-odds log(p / (1 - p)) for each row of X."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X must have shape (n_samples, {len(self.coef_)})")
        return X_arr @ self.coef_ + self.intercept_

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """P(y = 1) for each row of X."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray[np.int64]:
        """Class 1 where P(y = 1) >= threshold, else class 0."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(X) >= threshold).astype(np.int64)
