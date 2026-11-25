"""Tests for knn.py and data.py."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from data import CSV, SplitMix64, load_wine_split, standardize, synthetic, write_csv
from knn import KNNClassifier, Metric, Weights, nearest, pairwise_distances

POINTS = np.array([[2, 4], [0, 4], [5, 4], [0, 1], [3, 6], [2, 3], [5, 5]], float)
LABELS = np.array(["A", "A", "A", "A", "B", "B", "B"])
QUERY = np.array([[3.0, 3.0]])


def wine_scaled() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test, _ = load_wine_split()
    S_train, S_test = standardize(X_train, X_test)
    return S_train, S_test, y_train, y_test


class Distances(unittest.TestCase):
    def test_match_scipy(self) -> None:
        rng = np.random.default_rng(1)
        A, B = rng.normal(size=(300, 7)), rng.normal(size=(40, 7))
        assert_allclose(pairwise_distances(A, B), cdist(A, B), rtol=1e-12)
        assert_allclose(
            pairwise_distances(A, B, "manhattan"), cdist(A, B, "cityblock"), rtol=1e-12
        )

    def test_worked_example_distances(self) -> None:
        d = pairwise_distances(QUERY, POINTS)[0]
        assert_allclose(d**2, [2, 10, 5, 13, 9, 1, 8], rtol=1e-12)
        assert_allclose(
            pairwise_distances(QUERY, POINTS, "manhattan")[0], [2, 4, 3, 5, 3, 1, 4]
        )


class Neighbors(unittest.TestCase):
    def test_worked_example_predictions_change_with_k(self) -> None:
        got = [
            KNNClassifier(k=k).fit(POINTS, LABELS).predict(QUERY)[0]
            for k in (1, 3, 5, 7)
        ]
        self.assertEqual(got, ["B", "A", "B", "A"])

    def test_nearest_first_and_equal_distances_by_index(self) -> None:
        d = np.array([[3.0, 1.0, 2.0, 1.0, 0.5]])
        assert_array_equal(nearest(d, 4), [[4, 1, 3, 2]])

    def test_many_neighbors_come_back_sorted(self) -> None:
        # NumPy's partial selection happens to sort small k; large k exposes it
        rng = np.random.default_rng(4)
        X, Q = rng.random((2000, 3)), rng.random((20, 3))
        dist, idx = KNNClassifier(k=300).fit(X, np.zeros(2000)).kneighbors(Q)
        self.assertTrue((np.diff(dist, axis=1) >= 0).all())
        assert_array_equal(idx, np.argsort(cdist(Q, X), axis=1, kind="stable")[:, :300])

    def test_tied_vote_goes_to_the_smallest_label(self) -> None:
        X = np.array([[0.0], [2.0]])
        for y, want in (([5, 3], 3), ([3, 5], 3)):
            pred = KNNClassifier(k=2).fit(X, y).predict([[1.0]])
            self.assertEqual(pred[0], want)

    def test_distance_weights(self) -> None:
        X = np.array([[0.0], [1.0], [1.5]])
        model = KNNClassifier(k=3, weights="distance").fit(X, [0, 1, 1])
        # weights 1/0.2, 1/0.8 and 1/1.3: class 0 wins although class 1 has two votes
        self.assertEqual(model.predict([[0.2]])[0], 0)
        assert_allclose(
            model.predict_proba([[0.2]])[0],
            np.array([5.0, 1.25 + 1 / 1.3]) / (6.25 + 1 / 1.3),
        )
        # a neighbor at distance 0 takes the whole vote
        assert_allclose(model.predict_proba([[1.0]])[0], [0.0, 1.0])

    def test_one_neighbor_on_the_training_rows_is_always_right(self) -> None:
        S_train, _, y_train, _ = wine_scaled()
        pred = KNNClassifier(k=1).fit(S_train, y_train).predict(S_train)
        self.assertEqual(float((pred == y_train).mean()), 1.0)

    def test_probabilities_sum_to_one(self) -> None:
        S_train, S_test, y_train, _ = wine_scaled()
        for w in ("uniform", "distance"):
            p = (
                KNNClassifier(k=7, weights=w)
                .fit(S_train, y_train)
                .predict_proba(S_test)
            )
            assert_allclose(p.sum(axis=1), 1.0)


class AgainstScikitLearn(unittest.TestCase):
    def test_same_predictions_probabilities_and_neighbors(self) -> None:
        S_train, S_test, y_train, _ = wine_scaled()
        metrics: tuple[Metric, ...] = ("euclidean", "manhattan")
        weightings: tuple[Weights, ...] = ("uniform", "distance")
        for k in (1, 2, 5, 8, 27):
            for m in metrics:
                for w in weightings:
                    ours = KNNClassifier(k=k, metric=m, weights=w).fit(S_train, y_train)
                    sk = KNeighborsClassifier(
                        n_neighbors=k, metric=m, weights=w, algorithm="brute"
                    ).fit(S_train, y_train)
                    assert_array_equal(ours.predict(S_test), sk.predict(S_test))
                    assert_allclose(
                        ours.predict_proba(S_test), sk.predict_proba(S_test)
                    )
                    d_ours, i_ours = ours.kneighbors(S_test)
                    d_sk, i_sk = sk.kneighbors(S_test)
                    assert_array_equal(i_ours, i_sk)
                    assert_allclose(d_ours, d_sk, rtol=1e-10)

    def test_scaling_changes_the_wine_result(self) -> None:
        X_train, X_test, y_train, y_test, _ = load_wine_split()
        S_train, S_test = standardize(X_train, X_test)
        raw = KNNClassifier(k=5).fit(X_train, y_train).predict(X_test)
        scaled = KNNClassifier(k=5).fit(S_train, y_train).predict(S_test)
        self.assertEqual(int((raw == y_test).sum()), 35)
        self.assertEqual(int((scaled == y_test).sum()), 42)


class Data(unittest.TestCase):
    def test_split_sizes(self) -> None:
        X_train, X_test, y_train, _, names = load_wine_split()
        self.assertEqual((X_train.shape, X_test.shape), ((133, 13), (45, 13)))
        assert_array_equal(np.bincount(y_train), [44, 53, 36])
        self.assertEqual(names[-1], "proline")

    def test_standardize_uses_training_rows_only(self) -> None:
        rng = np.random.default_rng(3)
        A, B = rng.normal(5, 3, size=(50, 4)), rng.normal(100, 9, size=(10, 4))
        SA, SB = standardize(A, B)
        assert_allclose(SA.mean(axis=0), 0, atol=1e-12)
        assert_allclose(SA.std(axis=0), 1)
        assert_allclose(SB, (B - A.mean(axis=0)) / A.std(axis=0))

    def test_committed_csv_is_up_to_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "wine.csv"
            write_csv(fresh)
            self.assertEqual(fresh.read_bytes(), CSV.read_bytes())

    def test_splitmix64_reference_values(self) -> None:
        rng = SplitMix64(0)
        self.assertEqual(rng.next(), 0xE220A8397B1DCDAF)
        self.assertEqual(rng.next(), 0x6E789E6AA1B965F4)

    def test_synthetic_checksum_shared_with_cpp(self) -> None:
        # tests/test_knn.cpp in the C++ companion checks the same three numbers
        X, y = synthetic(2000, 8, seed=1)
        Q, _ = synthetic(200, 8, seed=2)
        pred = KNNClassifier(k=5).fit(X, y).predict(Q)
        self.assertEqual(X[0, 0], 0.5665615751722809)
        self.assertEqual(int(y.sum()), 1972)
        self.assertEqual(sum((i + 1) * int(p) for i, p in enumerate(pred)), 20892)

    def test_shared_output_has_no_near_ties(self) -> None:
        # the C++ program sums in a different order; a near-tie could swap neighbors
        X_train, X_test, _, _, _ = load_wine_split()
        S_train, S_test = standardize(X_train, X_test)
        for A, B in ((X_train, X_test), (S_train, S_test)):
            for m in ("euclidean", "manhattan"):
                d = np.sort(pairwise_distances(B, A, m), axis=1)
                for k in (1, 5, 7, 27):
                    self.assertGreater(float(np.min(d[:, k] - d[:, k - 1])), 1e-6)


class InputValidation(unittest.TestCase):
    def test_rejects_bad_input(self) -> None:
        X = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            KNNClassifier(k=4).fit(X, [0, 1, 2])
        with self.assertRaises(ValueError):
            KNNClassifier(k=1).fit(X, [0, 1])
        with self.assertRaises(ValueError):
            KNNClassifier(k=1).fit(np.array([[0.0, np.nan]]), [0])
        with self.assertRaises(ValueError):
            KNNClassifier(k=1).fit(X, [0, 1, 2]).predict(np.zeros((1, 3)))

    def test_rejects_bad_settings(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)
        with self.assertRaises(ValueError):
            KNNClassifier(metric="cosine")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            KNNClassifier(weights="inverse")  # type: ignore[arg-type]


class Programs(unittest.TestCase):
    def test_programs_print_the_expected_output(self) -> None:
        for name in ("worked_example", "wine_knn"):
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
