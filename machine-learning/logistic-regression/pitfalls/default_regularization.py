"""Pitfall: comparing a from-scratch model with scikit-learn's defaults.

LogisticRegression applies an L2 penalty unless told otherwise (C = 1.0), so
an unpenalized implementation "disagrees" with it even when both are right.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logistic_regression import LogisticRegressionScratch

hours = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).reshape(-1, 1)
passed = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])

ours = LogisticRegressionScratch().fit(hours, passed)
default = LogisticRegression().fit(hours, passed)
unpenalized = LogisticRegression(C=np.inf).fit(hours, passed)
matched = LogisticRegressionScratch(l2=1.0 / len(passed)).fit(hours, passed)

print(f"{'model':38} {'intercept':>9} {'weight':>7}")
print(f"{'from scratch, no penalty':38} {ours.intercept_:9.4f} {ours.coef_[0]:7.4f}")
print(
    f"{'scikit-learn, defaults (C=1.0)':38} {default.intercept_[0]:9.4f} {default.coef_[0, 0]:7.4f}"
)
print(
    f"{'scikit-learn, C=np.inf':38} {unpenalized.intercept_[0]:9.4f} {unpenalized.coef_[0, 0]:7.4f}"
)
print(
    f"{'from scratch, l2 = 1 / (C * n)':38} {matched.intercept_:9.4f} {matched.coef_[0]:7.4f}"
)
