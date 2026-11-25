"""Pitfall: reading raw-unit coefficients as feature importance."""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from linear_regression import LinearRegressionScratch

data = load_diabetes(scaled=False)
X, y, names = data.data, data.target, list(data.feature_names)

raw = LinearRegressionScratch().fit(X, y)
std = LinearRegressionScratch().fit(StandardScaler().fit_transform(X), y)

print(f"{'feature':8} {'raw units':>10} {'per std dev':>12}")
for i in np.argsort(-np.abs(raw.coef_)):
    print(f"{names[i]:8} {raw.coef_[i]:10.3f} {std.coef_[i]:12.2f}")

# Same model, BMI rescaled by 10: the BMI weight shrinks by 10,
# every prediction stays the same.
X10 = X.copy()
X10[:, 2] *= 10.0
rescaled = LinearRegressionScratch().fit(X10, y)
same = np.allclose(rescaled.predict(X10), raw.predict(X))
print(
    f"\nBMI weight: {raw.coef_[2]:.3f} -> {rescaled.coef_[2]:.4f} "
    f"after BMI x 10; predictions unchanged: {same}"
)
