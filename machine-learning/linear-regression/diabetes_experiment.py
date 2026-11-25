"""Linear regression on the diabetes dataset (442 patients, 10 features).

Compares a mean-predicting baseline, the from-scratch model fitted two ways,
and scikit-learn's LinearRegression on the same split, then repeats the split
100 times to show how much one test score can move.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from linear_regression import LinearRegressionScratch

# Raw units (age in years, BMI, blood pressure in mm Hg ...), not the
# pre-scaled copy, so that the scaler is fitted on training rows only.
data = load_diabetes(scaled=False)
X, y = data.data, data.target
names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

models = {
    "baseline (predict the mean)": DummyRegressor(strategy="mean"),
    "scratch, lstsq": LinearRegressionScratch(method="lstsq"),
    "scratch, gradient descent": LinearRegressionScratch(method="gd"),
    "scikit-learn LinearRegression": LinearRegression(),
}
print(f"train rows: {len(y_train)}, test rows: {len(y_test)}\n")
print(f"{'model':32} {'MAE':>7} {'RMSE':>7} {'R^2':>7}")
results = {}
for label, model in models.items():
    model.fit(Xs_train, y_train)
    pred = model.predict(Xs_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    results[label] = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"{label:32} {mae:7.2f} {rmse:7.2f} {r2:7.3f}")

scratch = models["scratch, lstsq"]
descent = models["scratch, gradient descent"]
sk = models["scikit-learn LinearRegression"]
assert isinstance(scratch, LinearRegressionScratch)
assert isinstance(descent, LinearRegressionScratch)
assert isinstance(sk, LinearRegression)
print(f"\ngradient descent: {descent.n_iter_} iterations")
print(
    f"largest coefficient gap, scratch lstsq vs scikit-learn: "
    f"{np.max(np.abs(scratch.coef_ - sk.coef_)):.1e}"
)
print(
    f"largest coefficient gap, gradient descent vs lstsq:     "
    f"{np.max(np.abs(descent.coef_ - scratch.coef_)):.1e}"
)

print("\ncoefficients (per standard deviation of each feature):")
order = np.argsort(-np.abs(scratch.coef_))
for i in order:
    print(f"  {names[i]:4} {scratch.coef_[i]:8.2f}")
print(f"  intercept {scratch.intercept_:.2f}")

# One split is one draw. Repeat with 100 different splits.
r2_linear, r2_base, coefs = [], [], []
for seed in range(100):
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler().fit(a)
    lin = LinearRegressionScratch().fit(sc.transform(a), c)
    r2_linear.append(r2_score(d, lin.predict(sc.transform(b))))
    coefs.append(lin.coef_)
    r2_base.append(r2_score(d, np.full_like(d, c.mean())))
lin_arr, base_arr = np.array(r2_linear), np.array(r2_base)
print("\ntest R^2 over 100 random 80/20 splits:")
print(
    f"  linear regression  mean {lin_arr.mean():.3f}  std {lin_arr.std():.3f}"
    f"  min {lin_arr.min():.3f}  max {lin_arr.max():.3f}"
)
print(f"  baseline           mean {base_arr.mean():.3f}  std {base_arr.std():.3f}")

# Correlated features share credit, so their weights move from split to split.
coef_arr = np.array(coefs)
corr = np.corrcoef(X_train, rowvar=False)
print("\ncoefficient spread over the same 100 splits:")
for name in ("s1", "s2", "bmi", "bp"):
    i = names.index(name)
    print(
        f"  {name:4} mean {coef_arr[:, i].mean():7.2f}  std {coef_arr[:, i].std():6.2f}"
        f"  range {coef_arr[:, i].min():7.2f} to {coef_arr[:, i].max():7.2f}"
    )
s1, s2 = names.index("s1"), names.index("s2")
print(f"correlation of s1 and s2 in the training rows: {corr[s1, s2]:.3f}")

out = {
    "split": {
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
    },
    "metrics": results,
    "gd_iterations": descent.n_iter_,
    "coef_gap_lstsq_vs_sklearn": float(np.max(np.abs(scratch.coef_ - sk.coef_))),
    "coef_gap_gd_vs_lstsq": float(np.max(np.abs(descent.coef_ - scratch.coef_))),
    "coefficients": dict(zip(names, map(float, scratch.coef_), strict=True)),
    "intercept": scratch.intercept_,
    "repeated_splits": {
        "n": 100,
        "r2_linear": r2_linear,
        "r2_baseline": r2_base,
        "coef_std": {n: float(coef_arr[:, i].std()) for i, n in enumerate(names)},
        "coef_min": {n: float(coef_arr[:, i].min()) for i, n in enumerate(names)},
        "coef_max": {n: float(coef_arr[:, i].max()) for i, n in enumerate(names)},
    },
    "corr_s1_s2_train": float(corr[s1, s2]),
}
Path("results").mkdir(exist_ok=True)
Path("results/diabetes.json").write_text(json.dumps(out, indent=2) + "\n")
