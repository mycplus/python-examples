"""Tests for tree.py, export.py and predict_exported.py."""

import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.tree import DecisionTreeClassifier as SklearnTree

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import export_model
from data import SplitMix64, benchmark_order, load_split
from export import save_tree, to_cpp
from predict_exported import load_tree, predict
from tree import Criterion, DecisionTreeClassifier, best_split, impurity

MONTHS = np.arange(1.0, 9.0).reshape(-1, 1)
CANCELLED = np.array([0, 0, 0, 0, 1, 0, 0, 1])
CRITERIA: tuple[Criterion, ...] = ("gini", "entropy")
FIRST_ROWS = [19, 83, 2, 30, 51]  # benchmark_order()[:5], also checked in C++
CHECKSUM = 732426313  # benchmark checksum, also printed by tree_bench in C++


def brute_force_gain(
    X: np.ndarray, y: np.ndarray, criterion: Criterion, min_leaf: int
) -> float:
    """The largest impurity decrease over every feature and threshold, by loops."""
    k = int(y.max()) + 1
    parent = float(impurity(np.bincount(y, minlength=k).astype(float), criterion))
    best = -np.inf
    for f in range(X.shape[1]):
        for t in np.unique(X[:, f])[:-1]:
            mask = X[:, f] <= t
            if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
                continue
            a = np.bincount(y[mask], minlength=k).astype(float)
            b = np.bincount(y[~mask], minlength=k).astype(float)
            n = len(y)
            child = (
                mask.sum() * impurity(a, criterion)
                + (~mask).sum() * impurity(b, criterion)
            ) / n
            best = max(best, parent - float(child))
    return best


class Impurity(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertAlmostEqual(float(impurity(np.array([6.0, 2.0]), "gini")), 0.375)
        self.assertAlmostEqual(
            float(impurity(np.array([6.0, 2.0]), "entropy")), 0.8112781244591328
        )
        self.assertEqual(float(impurity(np.array([5.0, 0.0]), "gini")), 0.0)
        self.assertEqual(float(impurity(np.array([0.0, 5.0]), "entropy")), 0.0)
        self.assertAlmostEqual(
            float(impurity(np.array([1.0, 1.0, 1.0]), "gini")), 2 / 3
        )
        self.assertAlmostEqual(
            float(impurity(np.array([1.0, 1.0, 1.0]), "entropy")), np.log2(3)
        )

    def test_worked_example_first_splits_differ(self) -> None:
        roots = [
            DecisionTreeClassifier(c, max_depth=1).fit(MONTHS, CANCELLED).threshold_[0]
            for c in CRITERIA
        ]
        self.assertEqual(roots, [7.5, 4.5])


class Splits(unittest.TestCase):
    def test_best_split_matches_brute_force(self) -> None:
        rng = np.random.default_rng(5)
        for trial in range(60):
            n, d, k = (
                int(rng.integers(2, 40)),
                int(rng.integers(1, 4)),
                int(rng.integers(2, 4)),
            )
            X = rng.integers(0, 6, size=(n, d)).astype(float)  # many equal values
            y = rng.integers(0, k, size=n)
            for criterion, min_leaf in itertools.product(CRITERIA, (1, 3)):
                split = best_split(X, y, k, criterion, min_leaf)
                want = brute_force_gain(X, y, criterion, min_leaf)
                if split is None:
                    self.assertEqual(want, -np.inf, (trial, criterion))
                else:
                    self.assertAlmostEqual(split.gain, want, places=12)

    def test_ties_go_to_the_lower_feature_then_the_lower_threshold(self) -> None:
        X = np.column_stack([MONTHS[:, 0], MONTHS[:, 0]])  # two identical features
        split = best_split(X, np.array([0, 0, 1, 1, 1, 1, 0, 0]), 2, "gini", 1)
        assert split is not None
        self.assertEqual((split.feature, split.threshold), (0, 2.5))  # 2.5 and 6.5 tie

    def test_threshold_is_halfway_between_neighbors(self) -> None:
        X = np.array([[1.0], [2.0], [10.0], [11.0]])
        tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
        self.assertEqual(tree.threshold_[0], 6.0)

    def test_a_value_equal_to_the_threshold_goes_left(self) -> None:
        X = np.array([[1.0], [2.0], [10.0], [11.0]])
        tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
        self.assertEqual(tree.predict([[6.0]])[0], 0)
        with tempfile.TemporaryDirectory() as tmp:
            save_tree(tree, Path(tmp) / "tree.txt")
            nodes = load_tree(Path(tmp) / "tree.txt")
        self.assertEqual(predict(nodes, [6.0])[0], 0)
        self.assertEqual(predict(nodes, [6.000000000000001])[0], 1)

    def test_neighboring_doubles_are_still_split(self) -> None:
        # halfway between two adjacent doubles rounds to the lower one
        X = np.array([[1.0], [np.nextafter(1.0, 2.0)]])
        tree = DecisionTreeClassifier().fit(X, [0, 1])
        self.assertEqual(tree.threshold_[0], 1.0)
        assert_array_equal(tree.predict(X), [0, 1])


class Growth(unittest.TestCase):
    def test_limits_are_respected(self) -> None:
        X_train, _, y_train, _, _ = load_split()
        for depth in (1, 2, 4):
            self.assertLessEqual(
                DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train).depth_,
                depth,
            )
        tree = DecisionTreeClassifier(min_samples_leaf=7).fit(X_train, y_train)
        leaves = tree.counts_[tree.feature_ < 0].sum(axis=1)
        self.assertGreaterEqual(int(leaves.min()), 7)
        tree = DecisionTreeClassifier(min_samples_split=40).fit(X_train, y_train)
        inner = tree.counts_[tree.feature_ >= 0].sum(axis=1)
        self.assertGreaterEqual(int(inner.min()), 40)

    def test_full_tree_fits_the_training_rows(self) -> None:
        X_train, _, y_train, _, _ = load_split()
        tree = DecisionTreeClassifier().fit(X_train, y_train)
        self.assertEqual(float((tree.predict(X_train) == y_train).mean()), 1.0)
        self.assertTrue(
            (tree.counts_[tree.feature_ < 0].min(axis=1) == 0).all()
        )  # pure

    def test_probabilities_are_leaf_fractions(self) -> None:
        X_train, X_test, y_train, _, _ = load_split()
        tree = DecisionTreeClassifier("entropy", max_depth=3).fit(X_train, y_train)
        p = tree.predict_proba(X_test)
        assert_allclose(p.sum(axis=1), 1.0)
        leaf = tree.apply(X_test[:1])[0]
        assert_allclose(p[0], tree.counts_[leaf] / tree.counts_[leaf].sum())

    def test_string_labels(self) -> None:
        tree = DecisionTreeClassifier().fit(
            MONTHS, np.where(CANCELLED == 1, "left", "stayed")
        )
        self.assertEqual(list(tree.predict([[8.0], [1.0]])), ["left", "stayed"])

    def test_scaling_changes_nothing_but_thresholds(self) -> None:
        X_train, X_test, y_train, _, _ = load_split()
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        raw = DecisionTreeClassifier().fit(X_train, y_train)
        scaled = DecisionTreeClassifier().fit((X_train - mean) / std, y_train)
        assert_array_equal(raw.feature_, scaled.feature_)
        assert_array_equal(raw.predict(X_test), scaled.predict((X_test - mean) / std))


class AgainstScikitLearn(unittest.TestCase):
    def test_identical_trees_when_no_splits_tie(self) -> None:
        # float32-exact values, because scikit-learn stores features as float32.
        # With one feature, or with leaves of at least 10 rows, no two splits
        # tie, so scikit-learn's random feature order cannot change its choice.
        rng = np.random.default_rng(11)
        X = rng.normal(size=(300, 5)).astype(np.float32).astype(np.float64)
        y = (X[:, 0] + X[:, 1] ** 2 + 0.5 * rng.normal(size=300) > 1).astype(int)
        for criterion in CRITERIA:
            for data, settings in (
                (X[:, :1], {}),
                (X, {"max_depth": 4, "min_samples_leaf": 10}),
            ):
                ours = DecisionTreeClassifier(criterion, **settings).fit(data, y)
                for seed in range(5):
                    sk = SklearnTree(criterion=criterion, random_state=seed, **settings)
                    t = sk.fit(data, y).tree_
                    inner = t.feature >= 0
                    assert_array_equal(
                        t.feature[inner], ours.feature_[ours.feature_ >= 0]
                    )
                    assert_array_equal(
                        t.threshold[inner], ours.threshold_[ours.feature_ >= 0]
                    )
                    assert_array_equal(t.n_node_samples, ours.counts_.sum(axis=1))

    def test_same_depth_3_entropy_tree_on_the_cancer_data(self) -> None:
        X_train, X_test, y_train, _, _ = load_split()
        ours = DecisionTreeClassifier("entropy", max_depth=3).fit(X_train, y_train)
        for seed in range(10):
            sk = SklearnTree(criterion="entropy", max_depth=3, random_state=seed)
            sk.fit(X_train, y_train)
            inner = sk.tree_.feature >= 0
            assert_array_equal(
                sk.tree_.feature[inner], ours.feature_[ours.feature_ >= 0]
            )
            # scikit-learn's thresholds come from float32 copies of the features
            assert_allclose(
                sk.tree_.threshold[inner],
                ours.threshold_[ours.feature_ >= 0],
                rtol=1e-6,
            )
            assert_array_equal(sk.predict(X_test), ours.predict(X_test))


class Export(unittest.TestCase):
    def test_saved_tree_and_generated_code_agree_with_the_model(self) -> None:
        X_train, X_test, y_train, _, names = load_split()
        tree = DecisionTreeClassifier("entropy", max_depth=3).fit(X_train, y_train)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.txt"
            save_tree(tree, path)
            nodes = load_tree(path)
        got = [predict(nodes, row)[0] for row in X_test.tolist()]
        assert_array_equal(got, tree.predict(X_test))
        source = to_cpp(tree, names, "f")
        self.assertEqual(source.count("return "), tree.n_leaves_)
        self.assertEqual(source.count("if (x["), tree.n_leaves_ - 1)

    def test_committed_model_files_are_up_to_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_model.main(Path(tmp))
            for name in (
                "models/tree.txt",
                "models/breast_cancer_tree.hpp",
                "data/breast_cancer_test.csv",
            ):
                self.assertEqual(
                    (Path(tmp) / name).read_bytes(), (HERE / name).read_bytes(), name
                )

    def test_benchmark_rows_and_checksum_shared_with_cpp(self) -> None:
        # the C++ tests and tree_bench in cpp-examples check the same numbers
        rng = SplitMix64(0)
        self.assertEqual(rng.next(), 0xE220A8397B1DCDAF)
        order = benchmark_order()
        self.assertEqual(order[:5].tolist(), FIRST_ROWS)
        X_train, X_test, y_train, _, _ = load_split()
        tree = DecisionTreeClassifier("entropy", max_depth=3).fit(X_train, y_train)
        pred = tree.predict(X_test[order]).astype(np.int64)
        weights = np.arange(1, len(pred) + 1, dtype=np.int64)
        self.assertEqual(int((weights * pred).sum() % 1_000_000_007), CHECKSUM)


class InputValidation(unittest.TestCase):
    def test_rejects_bad_input_and_settings(self) -> None:
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="log_loss")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_depth=0)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(min_samples_leaf=0)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(np.array([[np.nan]]), [0])
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(MONTHS, CANCELLED[:5])
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(MONTHS, CANCELLED).predict(np.zeros((1, 2)))


class Programs(unittest.TestCase):
    def test_programs_print_the_expected_output(self) -> None:
        for name in ("worked_example", "predict_exported"):
            out = subprocess.run(
                [sys.executable, f"{name}.py"],
                cwd=HERE,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            want = (HERE / "tests" / "expected" / f"{name}.txt").read_text()
            self.assertEqual(out.replace("\r\n", "\n"), want)


if __name__ == "__main__":
    unittest.main()
