"""The article's worked example: hours studied vs pass (1) or fail (0)."""

import numpy as np

from logistic_regression import (
    LogisticRegressionScratch,
    add_intercept,
    gradient,
    log_loss,
    sigmoid,
)

hours = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
passed = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
X = hours.reshape(-1, 1)
Xb = add_intercept(X)

# By hand at theta = 0: every probability is 0.5, the loss is ln 2.
theta0 = np.zeros(2)
print(
    f"at theta = 0: loss = {log_loss(Xb, passed, theta0):.4f} (ln 2 = {np.log(2):.4f})"
)
g = gradient(Xb, passed, theta0)
print(f"gradient: d/d(intercept) = {g[0]:.4f}, d/d(weight) = {g[1]:.4f}")

newton = LogisticRegressionScratch(method="newton").fit(X, passed)
descent = LogisticRegressionScratch(method="gd").fit(X, passed)
print(
    f"\nnewton: intercept = {newton.intercept_:.4f}, weight = {newton.coef_[0]:.4f}"
    f"  ({newton.n_iter_} iterations)"
)
print(
    f"gd:     intercept = {descent.intercept_:.4f}, weight = {descent.coef_[0]:.4f}"
    f"  ({descent.n_iter_} iterations)"
)
print(f"final loss = {newton.loss_history_[-1]:.4f}")

boundary = -newton.intercept_ / newton.coef_[0]
print(f"\nP(pass) = 0.5 at {boundary:.2f} hours")
print(
    f"each extra hour multiplies the odds of passing by {np.exp(newton.coef_[0]):.2f}"
)
print("\nhours  P(pass)")
for h in (1.0, 2.0, 3.0, 4.0, 5.0):
    p = sigmoid(np.array([newton.intercept_ + newton.coef_[0] * h]))[0]
    print(f"{h:5.1f}  {p:7.3f}")
