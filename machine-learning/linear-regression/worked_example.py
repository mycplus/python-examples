"""The article's worked example: five students, hours studied vs exam score."""

import numpy as np

from linear_regression import LinearRegressionScratch

hours = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
score = np.array([52.0, 55.0, 61.0, 64.0, 68.0])

# By hand: slope = Sxy / Sxx, intercept = mean(y) - slope * mean(x)
x_mean, y_mean = hours.mean(), score.mean()
s_xy = np.sum((hours - x_mean) * (score - y_mean))
s_xx = np.sum((hours - x_mean) ** 2)
slope = s_xy / s_xx
intercept = y_mean - slope * x_mean
print(f"By hand:  slope = {s_xy:.1f} / {s_xx:.1f} = {slope:.4f}")
print(
    f"          intercept = {y_mean:.1f} - {slope:.1f} * {x_mean:.1f} = {intercept:.4f}"
)

X = hours.reshape(-1, 1)  # one feature, so one column
exact = LinearRegressionScratch(method="lstsq").fit(X, score)
descent = LinearRegressionScratch(method="gd").fit(X, score)
print(f"lstsq:    slope = {exact.coef_[0]:.4f}, intercept = {exact.intercept_:.4f}")
print(
    f"gd:       slope = {descent.coef_[0]:.4f}, intercept = {descent.intercept_:.4f}"
    f"  ({descent.n_iter_} iterations, step {descent.learning_rate_:.5f})"
)

predicted = exact.predict(X)
residuals = score - predicted
print("\nhours  score  predicted  residual")
for x, y, p, r in zip(hours, score, predicted, residuals, strict=True):
    print(f"{x:5.0f}  {y:5.0f}  {p:9.1f}  {r:8.1f}")

sse = np.sum(residuals**2)
sst = np.sum((score - y_mean) ** 2)
print(f"\nSSE = {sse:.2f}, SST = {sst:.2f}, R^2 = 1 - SSE/SST = {1 - sse / sst:.4f}")
