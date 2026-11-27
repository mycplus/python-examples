"""Tests for the article's first-model script.

The point is not coverage for its own sake: these are the properties a reader is
told to expect, checked so that the prose cannot drift from the code.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from first_model import fit, load_split


class TestSplit(unittest.TestCase):
    def test_shapes(self) -> None:
        iris, X_train, X_test, y_train, y_test = load_split()
        self.assertEqual(X_train.shape, (112, 4))
        self.assertEqual(X_test.shape, (38, 4))
        self.assertEqual(y_train.shape, (112,))
        self.assertEqual(y_test.shape, (38,))
        self.assertEqual(len(iris.data), 150)

    def test_stratified(self) -> None:
        _, _, _, y_train, y_test = load_split()
        self.assertEqual(np.bincount(y_train).tolist(), [37, 37, 38])
        self.assertEqual(np.bincount(y_test).tolist(), [13, 13, 12])

    def test_split_is_a_partition(self) -> None:
        """Every row lands in exactly one side, checked as a multiset.

        Comparing row values instead would fail: iris contains one duplicated
        row, and the two copies can land on opposite sides of the split.
        """
        iris, X_train, X_test, _, _ = load_split()
        combined = np.sort(np.vstack([X_train, X_test]), axis=0)
        np.testing.assert_array_equal(combined, np.sort(iris.data, axis=0))

    def test_iris_has_a_duplicated_row(self) -> None:
        """149 unique rows out of 150: why overlap must be checked by index."""
        iris = load_split()[0]
        unique = {tuple(row) for row in iris.data}
        self.assertEqual(len(unique), 149)

    def test_same_seed_same_split(self) -> None:
        first, second = load_split(3), load_split(3)
        np.testing.assert_array_equal(first[1], second[1])

    def test_different_seed_different_split(self) -> None:
        first, second = load_split(0), load_split(1)
        self.assertFalse(np.array_equal(first[1], second[1]))


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        _, self.X_train, self.X_test, self.y_train, self.y_test = load_split()
        self.model = fit(self.X_train, self.y_train)

    def test_scaler_sees_training_rows_only(self) -> None:
        """The leak this article warns about, checked directly."""
        scaler = self.model.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, self.X_train.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, np.vstack([self.X_train, self.X_test]).mean(axis=0)))

    def test_predictions_are_deterministic(self) -> None:
        again = fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(self.model.predict(self.X_test), again.predict(self.X_test))

    def test_accuracy_beats_baseline(self) -> None:
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score

        baseline = DummyClassifier(strategy="most_frequent").fit(self.X_train, self.y_train)
        model_score = accuracy_score(self.y_test, self.model.predict(self.X_test))
        baseline_score = accuracy_score(self.y_test, baseline.predict(self.X_test))
        self.assertGreater(model_score, baseline_score)

    def test_confusion_matrix_sums_to_test_rows(self) -> None:
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(int(matrix.sum()), len(self.y_test))

    def test_known_accuracy(self) -> None:
        from sklearn.metrics import accuracy_score

        self.assertAlmostEqual(
            accuracy_score(self.y_test, self.model.predict(self.X_test)), 0.9737, places=4
        )

    def test_unseen_scale_still_predicts(self) -> None:
        """Ten times the training range must not raise, whatever it predicts."""
        predicted = self.model.predict(self.X_test * 10)
        self.assertEqual(predicted.shape, (38,))
        self.assertTrue(set(predicted).issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
