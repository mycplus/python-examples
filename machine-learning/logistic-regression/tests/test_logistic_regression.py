"""Tests for logistic_regression.py."""

import itertools
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logistic_regression import (
    LogisticRegressionScratch,
    add_intercept,
    gradient,
    hessian,
    lipschitz_constant,
    log_loss,
    sigmoid,
)

HOURS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).reshape(-1, 1)
PASSED = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])


def cancer() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X), 1.0 - y


class Sigmoid(unittest.TestCase):
    def test_extreme_inputs_do_not_overflow(self) -> None:
        z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        with np.errstate(over="raise", invalid="raise"):
            p = sigmoid(z)
        assert_allclose(p, [0.0, 1.9287e-22, 0.5, 1.0, 1.0], rtol=1e-4, atol=1e-300)

    def test_symmetry(self) -> None:
        z = np.linspace(-30, 30, 61)
        assert_allclose(sigmoid(z) + sigmoid(-z), 1.0, rtol=1e-15)


class LossAndDerivatives(unittest.TestCase):
    def test_loss_at_zero_is_ln2(self) -> None:
        Xb = add_intercept(HOURS)
        self.assertAlmostEqual(
            log_loss(Xb, PASSED, np.zeros(2)), np.log(2.0), places=12
        )

    def test_gradient_at_zero_matches_hand_calculation(self) -> None:
        g = gradient(add_intercept(HOURS), PASSED, np.zeros(2))
        assert_allclose(g, [0.0, -0.475], atol=1e-12)

    def test_gradient_and_hessian_match_finite_differences(self) -> None:
        X, y = cancer()
        Xb = add_intercept(X[:, :5])
        theta = np.random.default_rng(3).normal(size=6) * 0.3
        h = 1e-6
        eye = np.eye(6)
        num_g = np.array(
            [
                (
                    log_loss(Xb, y, theta + h * e, 0.1)
                    - log_loss(Xb, y, theta - h * e, 0.1)
                )
                / (2 * h)
                for e in eye
            ]
        )
        assert_allclose(gradient(Xb, y, theta, 0.1), num_g, atol=1e-7)
        num_h = np.array(
            [
                (
                    gradient(Xb, y, theta + h * e, 0.1)
                    - gradient(Xb, y, theta - h * e, 0.1)
                )
                / (2 * h)
                for e in eye
            ]
        )
        assert_allclose(hessian(Xb, theta, 0.1), num_h, atol=1e-6)


class WorkedExample(unittest.TestCase):
    def test_newton_matches_unpenalized_scikit_learn(self) -> None:
        ours = LogisticRegressionScratch().fit(HOURS, PASSED)
        theirs = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(
            HOURS, PASSED
        )
        assert_allclose(
            [ours.intercept_, ours.coef_[0]],
            [theirs.intercept_[0], theirs.coef_[0, 0]],
            rtol=1e-5,
        )
        assert_allclose([ours.intercept_, ours.coef_[0]], [-3.7219, 1.3534], atol=1e-4)

    def test_gradient_descent_agrees_with_newton(self) -> None:
        a = LogisticRegressionScratch(method="newton").fit(HOURS, PASSED)
        b = LogisticRegressionScratch(method="gd").fit(HOURS, PASSED)
        assert_allclose(b.coef_, a.coef_, rtol=1e-5)
        self.assertAlmostEqual(b.intercept_, a.intercept_, places=4)

    def test_iteration_counts(self) -> None:
        # A gradient off by a constant factor still converges, only more
        # slowly, so check the work done as well as the answer.
        newton = LogisticRegressionScratch(method="newton").fit(HOURS, PASSED)
        descent = LogisticRegressionScratch(method="gd").fit(HOURS, PASSED)
        self.assertLessEqual(newton.n_iter_, 8)
        self.assertGreaterEqual(descent.n_iter_, 2200)
        self.assertLessEqual(descent.n_iter_, 2800)

    def test_decision_boundary(self) -> None:
        m = LogisticRegressionScratch().fit(HOURS, PASSED)
        boundary = -m.intercept_ / m.coef_[0]
        self.assertAlmostEqual(float(m.predict_proba([[boundary]])[0]), 0.5, places=12)
        np.testing.assert_array_equal(m.predict([[2.7], [2.8]]), [0, 1])


class RealData(unittest.TestCase):
    def test_matches_scikit_learn_with_the_same_penalty(self) -> None:
        X, y = cancer()
        C = 1.0
        ours = LogisticRegressionScratch(l2=1.0 / (C * len(y))).fit(X, y)
        theirs = LogisticRegression(
            C=C, solver="newton-cholesky", tol=1e-12, max_iter=1000
        ).fit(X, y)
        assert_allclose(ours.coef_, theirs.coef_[0], atol=1e-6)
        self.assertAlmostEqual(ours.intercept_, float(theirs.intercept_[0]), places=6)

    def test_gradient_descent_matches_newton_with_penalty(self) -> None:
        X, y = cancer()
        a = LogisticRegressionScratch(method="newton", l2=0.01).fit(X, y)
        b = LogisticRegressionScratch(method="gd", l2=0.01).fit(X, y)
        assert_allclose(b.coef_, a.coef_, atol=1e-5)

    def test_gradient_descent_loss_never_increases(self) -> None:
        X, y = cancer()
        h = np.array(
            LogisticRegressionScratch(method="gd", l2=0.01).fit(X, y).loss_history_
        )
        self.assertTrue(np.all(np.diff(h) <= 1e-12 * h[:-1]))

    def test_optimality_condition(self) -> None:
        X, y = cancer()
        m = LogisticRegressionScratch(l2=0.01).fit(X, y)
        theta = np.concatenate([[m.intercept_], m.coef_])
        self.assertLess(
            np.linalg.norm(gradient(add_intercept(X), y, theta, 0.01)), 1e-9
        )


class Separation(unittest.TestCase):
    def test_separable_data_without_penalty_never_converges(self) -> None:
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = LogisticRegressionScratch(method="gd", max_iter=20_000).fit(X, y)
        self.assertFalse(m.converged_)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        norms = m.weight_norm_history_
        self.assertTrue(all(b > a for a, b in itertools.pairwise(norms[1:])))

    def test_penalty_makes_separable_data_converge(self) -> None:
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        for method in ("newton", "gd"):
            with self.subTest(method=method):
                m = LogisticRegressionScratch(method=method, l2=0.1).fit(X, y)
                self.assertTrue(m.converged_)
                self.assertLess(abs(m.coef_[0]), 10.0)


class Steps(unittest.TestCase):
    def test_default_step_is_one_over_lipschitz_constant(self) -> None:
        m = LogisticRegressionScratch(method="gd").fit(HOURS, PASSED)
        assert m.learning_rate_ is not None
        self.assertAlmostEqual(
            m.learning_rate_, 1.0 / lipschitz_constant(add_intercept(HOURS))
        )

    def test_too_large_a_step_makes_the_loss_go_up(self) -> None:
        # The log loss grows only linearly in the weights, so a bad step does
        # not overflow; it oscillates. 1/L is the step that cannot do this.
        X, y = cancer()
        step = 50.0 / lipschitz_constant(add_intercept(X), 0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = LogisticRegressionScratch(
                method="gd", l2=0.01, learning_rate=step, max_iter=200
            ).fit(X, y)
        self.assertTrue(np.any(np.diff(m.loss_history_) > 0))

    def test_is_deterministic(self) -> None:
        X, y = cancer()
        a = LogisticRegressionScratch(method="gd", l2=0.01).fit(X, y)
        b = LogisticRegressionScratch(method="gd", l2=0.01).fit(X, y)
        np.testing.assert_array_equal(a.coef_, b.coef_)


class InputValidation(unittest.TestCase):
    def test_rejects_bad_input(self) -> None:
        cases: dict[str, tuple[ArrayLike, ArrayLike, str]] = {
            "1-D X": ([1.0, 2.0], [0.0, 1.0], "2-D"),
            "length mismatch": ([[1.0], [2.0]], [0.0, 1.0, 1.0], "labels"),
            "empty": (np.empty((0, 1)), np.empty(0), "zero samples"),
            "NaN": ([[1.0], [np.nan]], [0.0, 1.0], "NaN"),
            "labels 1 and 2": ([[1.0], [2.0]], [1.0, 2.0], "0 and 1"),
        }
        for label, (X, y, message) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                LogisticRegressionScratch().fit(X, y)

    def test_rejects_bad_settings(self) -> None:
        with self.assertRaises(ValueError):
            LogisticRegressionScratch(method="sgd")
        with self.assertRaises(ValueError):
            LogisticRegressionScratch(l2=-1.0)
        m = LogisticRegressionScratch().fit(HOURS, PASSED)
        with self.assertRaises(ValueError):
            m.predict(HOURS, threshold=1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            m.predict_proba([[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
