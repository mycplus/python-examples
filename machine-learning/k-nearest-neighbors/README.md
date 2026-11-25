# K-Nearest Neighbors (KNN) in Python From Scratch

[![k-nearest-neighbors](https://github.com/mycplus/python-examples/actions/workflows/k-nearest-neighbors.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/k-nearest-neighbors.yml)

Companion code for [K-Nearest Neighbors (KNN): How It Works, With Python and C++](https://www.mycplus.com/computer-science/ai/k-nearest-neighbors/) on MYCPLUS. The C++ version is in [cpp-examples/machine-learning/k-nearest-neighbors](https://github.com/mycplus/cpp-examples/tree/main/machine-learning/k-nearest-neighbors); both read `data/wine_split.csv` and print the same `tests/expected/wine_knn.txt`.

| File | What it is |
| --- | --- |
| `knn.py` | `KNNClassifier`: Euclidean or Manhattan distance, uniform or distance-weighted votes, NumPy only |
| `data.py` | The shared wine split (75/25, stratified), standardization on the training rows, and a splitmix64 data generator that the C++ program reproduces |
| `worked_example.py` | The article's seven-point example: one query, four values of k |
| `wine_knn.py` | Reads `data/wine_split.csv` and prints the output the C++ program must match |
| `wine_experiment.py` | Unscaled vs standardized, k, distance and vote weights chosen by repeated cross-validation; checks scikit-learn makes the same predictions |
| `dimensionality_experiment.py` | Distance concentration in random data, and noise columns added to the wine data |
| `benchmark.py` | Pure Python, this NumPy code and scikit-learn timed on the same 20,000-row synthetic data, one thread (about 30 seconds) |
| `pitfalls/optimistic_evaluation.py` | Scoring on the training rows, and choosing k on the test rows |
| `mutations.py` | Re-applies 16 plausible bugs to a copy of the code and checks the tests catch each (about 1 minute) |
| `tests/` | `unittest` tests and the expected output of three programs |
| `results/` | Outputs captured for the article, plus the data behind its figures |

## Data

The [Wine](https://archive.ics.uci.edu/dataset/109/wine) data (178 wines from three cultivars, 13 chemical measurements) ships with scikit-learn as `load_wine`. Aeberhard, S. and Forina, M. (1992), UCI Machine Learning Repository, [doi:10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), licensed CC BY 4.0. `data/wine_split.csv` is the same data with the article's train/test split; `python -c "import data; data.write_csv()"` regenerates it, and a test checks the committed file is current. Nothing is downloaded.

## Run and test

Python 3.13 or later.

```sh
python -m pip install -r requirements-dev.txt
python worked_example.py
python -m unittest discover -s tests -v
sh run_all.sh          # every check, and regenerates results/
```

On Windows, `.\run_all.ps1` runs the same checks and writes `results-windows.md`.

## What the build checks

- Runs on Python 3.13, 3.14 and 3.15, on Linux and Windows, with the pinned versions in `requirements.txt`.
- `ruff check`, `ruff format --check` and `mypy --strict` report nothing.
- 20 tests: distances against SciPy, the worked example, neighbor order and tie rules, distance-weighted votes, identical predictions, probabilities, neighbors and distances to scikit-learn on the wine data, standardization on training rows only, the committed CSV, the splitmix64 stream and a checksum shared with the C++ tests, and input validation.
- `worked_example.py`, `wine_knn.py` and `wine_experiment.py` print exactly the output shown in the article.
- `mutations.py` (Linux, Python 3.15): all 16 deliberate bugs make a test fail.

Not checked in CI: `benchmark.py`, whose timings depend on the machine.
