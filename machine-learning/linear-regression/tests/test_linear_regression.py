"""Tests for linear_regression.py."""

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from linear_regression import (
    LinearRegressionScratch,
    add_intercept,
    lipschitz_constant,
)

HOURS = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
SCORES = np.array([52.0, 55.0, 61.0, 64.0, 68.0])


def diabetes() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_diabetes(scaled=False, return_X_y=True)
    return X, y


class WorkedExample(unittest.TestCase):
    def test_lstsq_matches_hand_calculation(self) -> None:
        model = LinearRegressionScratch().fit(HOURS, SCORES)
        self.assertAlmostEqual(model.coef_[0], 4.1, places=10)
        self.assertAlmostEqual(model.intercept_, 47.7, places=10)

    def test_gradient_descent_matches_lstsq(self) -> None:
        model = LinearRegressionScratch(method="gd").fit(HOURS, SCORES)
        self.assertTrue(model.converged_)
        assert_allclose([model.intercept_, model.coef_[0]], [47.7, 4.1], rtol=1e-6)

    def test_gradient_descent_takes_the_expected_number_of_steps(self) -> None:
        # Results alone cannot catch a gradient that is off by a constant
        # factor: it still converges, only more slowly. Check the work done.
        model = LinearRegressionScratch(method="gd").fit(HOURS, SCORES)
        self.assertGreaterEqual(model.n_iter_, 1100)
        self.assertLessEqual(model.n_iter_, 1300)


class Correctness(unittest.TestCase):
    def test_recovers_noiseless_coefficients(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = 3.0 + X @ np.array([1.5, -2.0, 0.0, 4.0])
        for method in ("lstsq", "gd"):
            with self.subTest(method=method):
                model = LinearRegressionScratch(method=method).fit(X, y)
                assert_allclose(model.coef_, [1.5, -2.0, 0.0, 4.0], atol=1e-6)
                self.assertAlmostEqual(model.intercept_, 3.0, places=6)

    def test_matches_scikit_learn_on_diabetes(self) -> None:
        X, y = diabetes()
        ours = LinearRegressionScratch().fit(X, y)
        theirs = LinearRegression().fit(X, y)
        assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8)
        self.assertAlmostEqual(ours.intercept_, float(theirs.intercept_), places=6)

    def test_gradient_descent_matches_lstsq_on_standardized_diabetes(self) -> None:
        X, y = diabetes()
        Xs = StandardScaler().fit_transform(X)
        exact = LinearRegressionScratch().fit(Xs, y)
        descent = LinearRegressionScratch(method="gd").fit(Xs, y)
        assert_allclose(descent.coef_, exact.coef_, atol=1e-3)
        assert_allclose(descent.predict(Xs), exact.predict(Xs), atol=1e-4)

    def test_residuals_are_orthogonal_to_every_column(self) -> None:
        # The normal equations X^T (y - X theta) = 0 hold at the solution.
        X, y = diabetes()
        model = LinearRegressionScratch().fit(X, y)
        Xb = add_intercept(X)
        residuals = y - model.predict(X)
        scale = np.abs(Xb).T @ np.abs(y)
        self.assertTrue(np.all(np.abs(Xb.T @ residuals) <= 1e-10 * scale))

    def test_reported_loss_is_the_mean_squared_error(self) -> None:
        for method in ("lstsq", "gd"):
            with self.subTest(method=method):
                model = LinearRegressionScratch(method=method).fit(HOURS, SCORES)
                expected = np.mean((model.predict(HOURS) - SCORES) ** 2)
                self.assertAlmostEqual(model.loss_history_[-1], expected, places=9)
                self.assertAlmostEqual(model.loss_history_[-1], 1.9 / 5, places=6)


class GradientDescent(unittest.TestCase):
    def test_loss_never_increases_with_the_default_step(self) -> None:
        X, y = diabetes()
        Xs = StandardScaler().fit_transform(X)
        history = np.array(
            LinearRegressionScratch(method="gd").fit(Xs, y).loss_history_
        )
        self.assertTrue(np.all(np.diff(history) <= 1e-12 * history[:-1]))

    def test_default_step_is_one_over_lipschitz_constant(self) -> None:
        model = LinearRegressionScratch(method="gd").fit(HOURS, SCORES)
        self.assertAlmostEqual(
            model.learning_rate_, 1.0 / lipschitz_constant(add_intercept(HOURS))
        )

    def test_too_large_a_step_diverges_with_a_clear_error(self) -> None:
        step = 2.5 / lipschitz_constant(add_intercept(HOURS))
        model = LinearRegressionScratch(method="gd", learning_rate=step)
        with self.assertRaisesRegex(FloatingPointError, "diverged"):
            model.fit(HOURS, SCORES)

    def test_warns_when_it_runs_out_of_iterations(self) -> None:
        X, y = diabetes()
        model = LinearRegressionScratch(method="gd", max_iter=50)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
        self.assertFalse(model.converged_)
        self.assertEqual(model.n_iter_, 50)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_is_deterministic(self) -> None:
        X, y = diabetes()
        Xs = StandardScaler().fit_transform(X)
        a = LinearRegressionScratch(method="gd").fit(Xs, y)
        b = LinearRegressionScratch(method="gd").fit(Xs, y)
        np.testing.assert_array_equal(a.coef_, b.coef_)
        self.assertEqual(a.n_iter_, b.n_iter_)


class EdgeCases(unittest.TestCase):
    def test_duplicated_column_gives_the_same_predictions(self) -> None:
        X, y = diabetes()
        X_dup = np.column_stack([X, X[:, 2] * 10.0])
        base = LinearRegressionScratch().fit(X, y)
        dup = LinearRegressionScratch().fit(X_dup, y)
        assert_allclose(dup.predict(X_dup), base.predict(X), rtol=1e-8)

    def test_more_features_than_samples_interpolates(self) -> None:
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(3, 5)), rng.normal(size=3)
        model = LinearRegressionScratch().fit(X, y)
        assert_allclose(model.predict(X), y, atol=1e-10)

    def test_single_sample(self) -> None:
        model = LinearRegressionScratch().fit([[2.0]], [7.0])
        self.assertAlmostEqual(float(model.predict([[2.0]])[0]), 7.0)

    def test_constant_feature(self) -> None:
        X = np.column_stack([np.arange(6.0), np.full(6, 3.0)])
        y = 1.0 + 2.0 * np.arange(6.0)
        model = LinearRegressionScratch().fit(X, y)
        assert_allclose(model.predict(X), y, atol=1e-10)

    def test_rescaling_a_feature_rescales_its_weight_only(self) -> None:
        X, y = diabetes()
        X10 = X.copy()
        X10[:, 2] *= 10.0
        a = LinearRegressionScratch().fit(X, y)
        b = LinearRegressionScratch().fit(X10, y)
        self.assertAlmostEqual(b.coef_[2] * 10.0, a.coef_[2], places=8)
        assert_allclose(b.predict(X10), a.predict(X), rtol=1e-10)


class InputValidation(unittest.TestCase):
    def test_rejects_bad_input(self) -> None:
        cases: dict[str, tuple[ArrayLike, ArrayLike, str]] = {
            "1-D X": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "2-D"),
            "2-D y": ([[1.0], [2.0]], [[1.0], [2.0]], "1-D"),
            "length mismatch": ([[1.0], [2.0]], [1.0, 2.0, 3.0], "2 rows"),
            "empty": (np.empty((0, 2)), np.empty(0), "zero samples"),
            "NaN": ([[1.0], [np.nan]], [1.0, 2.0], "NaN"),
        }
        for label, (X, y, message) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                LinearRegressionScratch().fit(X, y)

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            LinearRegressionScratch(method="newton")

    def test_predict_checks_the_number_of_features(self) -> None:
        model = LinearRegressionScratch().fit(HOURS, SCORES)
        with self.assertRaisesRegex(ValueError, "shape"):
            model.predict([[1.0, 2.0]])


class Pitfalls(unittest.TestCase):
    def test_collinear_columns_make_the_normal_equations_unreliable(self) -> None:
        # What np.linalg.solve does with a singular X^T X depends on the
        # machine: it has raised LinAlgError, returned weights near 1e13, and
        # returned ordinary-looking weights. So the test checks only what holds
        # everywhere: X^T X is numerically singular, and lstsq reports the true
        # rank and gives the predictions of the model without the copy.
        X, y = diabetes()
        Xb = np.column_stack([np.ones(len(y)), X, X[:, 2] * 10.0])
        self.assertGreater(np.linalg.cond(Xb.T @ Xb), 1e15)
        theta, _, rank, _ = np.linalg.lstsq(Xb, y)
        self.assertEqual(rank, Xb.shape[1] - 1)
        base = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y)[0]
        assert_allclose(
            Xb @ theta, np.column_stack([np.ones(len(y)), X]) @ base, rtol=1e-8
        )
        # The two BMI weights add up to the single-column weight.
        self.assertAlmostEqual(theta[3] + 10.0 * theta[-1], base[3], places=6)


if __name__ == "__main__":
    unittest.main()
