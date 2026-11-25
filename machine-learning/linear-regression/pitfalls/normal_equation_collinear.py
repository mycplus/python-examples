"""Pitfall: solving the normal equations when two columns are collinear.

The BMI column is added a second time, multiplied by 10, as happens when one
quantity arrives in two units. X^T X is then singular in exact arithmetic.
"""

import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(scaled=False, return_X_y=True)
bmi = X[:, 2]
X_dup = np.column_stack([np.ones(len(y)), X, bmi * 10.0])

gram = X_dup.T @ X_dup
print(f"columns: {X_dup.shape[1]}, rank of X: {np.linalg.matrix_rank(X_dup)}")
print(f"condition number of X^T X: {np.linalg.cond(gram):.1e}")

try:
    theta_ne = np.linalg.solve(gram, X_dup.T @ y)
    pred_ne = X_dup @ theta_ne
    print("normal equations: solve() returned without an error")
    print(f"  bmi weights: {theta_ne[3]:.4g} and {theta_ne[-1]:.4g}")
    print(f"  training MSE: {np.mean((pred_ne - y) ** 2):.2f}")
except np.linalg.LinAlgError as err:
    print(f"normal equations: LinAlgError: {err}")

theta_ls, _, rank, _ = np.linalg.lstsq(X_dup, y)
pred_ls = X_dup @ theta_ls
print(f"lstsq: rank {rank}, bmi weights {theta_ls[3]:.4g} and {theta_ls[-1]:.4g}")
print(f"  training MSE: {np.mean((pred_ls - y) ** 2):.2f}")

X_one = np.column_stack([np.ones(len(y)), X])
theta_one = np.linalg.lstsq(X_one, y)[0]
print(
    f"without the copy: bmi weight {theta_one[3]:.4g}, "
    f"training MSE {np.mean((X_one @ theta_one - y) ** 2):.2f}"
)
