# Linear Regression in Python From Scratch

[![linear-regression](https://github.com/mycplus/python-examples/actions/workflows/linear-regression.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/linear-regression.yml)

Companion code for [Linear Regression in Python From Scratch](https://www.mycplus.com/computer-science/ai/linear-regression/) on MYCPLUS.

| File | What it is |
| --- | --- |
| `linear_regression.py` | `LinearRegressionScratch`: least squares (`method="lstsq"`) and batch gradient descent (`method="gd"`), NumPy only |
| `worked_example.py` | The article's five-point example, by hand and with both methods |
| `diabetes_experiment.py` | Baseline, both scratch fits and scikit-learn on the diabetes data; 100 repeated splits |
| `scaling_experiment.py` | Gradient descent on raw vs standardized features (about 15 seconds) |
| `pitfalls/` | Three small programs: collinear normal equations, a missing intercept, raw-unit coefficients |
| `mutations.py` | Re-applies 11 plausible bugs to a copy of the code and checks the tests catch each |
| `tests/` | `unittest` tests and the expected output of three programs |
| `results/` | Outputs captured for the article (`*.txt`), plus the data behind its figures |

## Data

The [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) ships with scikit-learn (442 patients, 10 baseline features, disease progression after one year). It comes from Efron, Hastie, Johnstone and Tibshirani, "Least Angle Regression", *Annals of Statistics* 32(2), 2004. The code loads it with `scaled=False` so that scaling is fitted on training rows only. Nothing is downloaded.

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
- 22 tests: the hand-computed example, agreement with scikit-learn on the diabetes data, gradient descent agreeing with least squares, the normal equations holding at the solution, the loss never increasing, a clear error when the step is too large, a warning when gradient descent runs out of iterations, collinear and constant columns, more features than samples, and input validation.
- Three programs print exactly the output shown in the article.
- `mutations.py` (Linux, Python 3.15): all 11 deliberate bugs make a test fail.

Not checked in CI: the timings in `scaling_experiment.py`, which depend on the machine.
