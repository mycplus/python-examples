"""Pitfall: fitting least squares without an intercept column."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(scaled=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

# Wrong: no column of ones, so the fitted line must pass through the origin.
w = np.linalg.lstsq(Xs_train, y_train)[0]
print(f"without intercept: test R^2 = {r2_score(y_test, Xs_test @ w):.3f}")

# Right: prepend the column of ones.
ones_train = np.column_stack([np.ones(len(y_train)), Xs_train])
ones_test = np.column_stack([np.ones(len(y_test)), Xs_test])
theta = np.linalg.lstsq(ones_train, y_train)[0]
print(
    f"with intercept:    test R^2 = {r2_score(y_test, ones_test @ theta):.3f}"
    f"  (intercept {theta[0]:.2f}, mean of y_train {y_train.mean():.2f})"
)
