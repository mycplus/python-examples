# What Is Machine Learning?

[![what-is-machine-learning](https://github.com/mycplus/python-examples/actions/workflows/what-is-machine-learning.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/what-is-machine-learning.yml)

Companion code for [What Is Machine Learning? Types, Workflow and a First Model in Python](https://www.mycplus.com/computer-science/ai/what-is-machine-learning/) on MYCPLUS. The first model is 39 lines on purpose, and the other two scripts show what the same data looks like under the other two learning paradigms.

| File | What it is |
| --- | --- |
| `first_model.py` | 39 lines: stratified 75/25 split of the iris data, `StandardScaler` inside a pipeline, logistic regression, majority-class baseline, accuracy and confusion matrix |
| `paradigms.py` | The same 150 rows clustered with k-means and no labels (silhouette, adjusted Rand index against the species), then tabular Q-learning on a 4×4 gridworld with two holes |
| `learning_curve.py` | The same pipeline on 10 fractions of the training rows, 20 split seeds each, written to `results/learning_curve.json` |
| `tests/` | 12 `unittest` tests and the expected output of the three programs |
| `results/` | The JSON files behind the article's learning-curve figure and its hero |

## Data

Iris ships with scikit-learn as `load_iris` (150 rows, 4 features, 3 balanced classes). Fisher (1936) and the UCI Machine Learning Repository; distributed inside scikit-learn under its BSD-3-Clause licence. Nothing is downloaded. One row of the 150 is a duplicate, and `tests/test_first_model.py` asserts that, because it is why the split is checked as a partition rather than by comparing row values.

The gridworld is defined in `paradigms.py`: 4×4, goal at (3, 3), holes at (1, 1) and (2, 2), reward 1.0 for the goal, −1.0 for a hole and −0.01 per step.

## Run and test

Python 3.13 or later.

```sh
python -m pip install -r requirements-dev.txt
python first_model.py
python -m unittest discover -s tests -v
```

`python paradigms.py` and `python learning_curve.py` regenerate the other two sets of numbers. The three scripts are deterministic; the learning curve repeats 10 fractions across 20 split seeds and writes `results/learning_curve.json`.

## What the build checks

- Runs on Python 3.13, 3.14 and 3.15, on Linux and Windows, with the pinned versions in `requirements.txt`.
- `ruff check` is clean. Formatting is not enforced, because `ruff format` would rewrite `first_model.py` onto 40 lines and the 39-line count is the point of the example.
- The 12 tests pass, including the two claims the article leans on: the scaler is fitted on training rows only, and the split is a partition of the 150 rows.
- `first_model.py`, `paradigms.py` and `learning_curve.py` print exactly the output committed in `tests/expected/`.

The three scripts are deliberately unannotated: the point of `first_model.py` is that a beginner can read all 39 lines, so `mypy --strict` is not applied to this directory the way it is to the algorithm articles.
