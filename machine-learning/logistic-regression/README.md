# Logistic Regression in Python From Scratch

[![logistic-regression](https://github.com/mycplus/python-examples/actions/workflows/logistic-regression.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/logistic-regression.yml)

Companion code for [Logistic Regression in Python From Scratch](https://www.mycplus.com/computer-science/ai/logistic-regression/) on MYCPLUS.

| File | What it is |
| --- | --- |
| `logistic_regression.py` | `LogisticRegressionScratch`: Newton's method (`method="newton"`) and batch gradient descent (`method="gd"`) on the log loss, optional L2 penalty, NumPy only |
| `worked_example.py` | The article's ten-student example: the loss and gradient by hand, then both fits |
| `data.py` | The shared breast cancer split (80/20, stratified, malignant = 1) |
| `breast_cancer_experiment.py` | Baseline, both scratch fits and scikit-learn with the same penalty; confusion matrix and odds ratios |
| `threshold_experiment.py` | Picks the decision threshold from cross-validated training predictions, then applies it to the test rows |
| `separation_experiment.py` | What happens with no penalty on separable data (about 1 minute) |
| `pitfalls/default_regularization.py` | scikit-learn's default C = 1.0 vs an unpenalized model |
| `mutations.py` | Re-applies 11 plausible bugs to a copy of the code and checks the tests catch each (about 6 minutes) |
| `tests/` | `unittest` tests and the expected output of two programs |
| `results/` | Outputs captured for the article, plus the data behind its figures |

## Data

The [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) data (569 tumours, 30 features computed from images of fine-needle aspirates) ships with scikit-learn as `load_breast_cancer`. Creators: W. Wolberg, W. N. Street, O. Mangasarian; UCI Machine Learning Repository, [doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), licensed CC BY 4.0. scikit-learn labels malignant as 0; this code flips the labels so malignant = 1. Nothing is downloaded.

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
- 20 tests: a stable sigmoid, the loss and gradient at zero by hand, gradient and Hessian against finite differences, agreement with scikit-learn (unpenalized and with the same penalty), Newton and gradient descent agreeing, iteration counts, the loss never increasing with the default step, separable data never converging without a penalty and converging with one, and input validation.
- `worked_example.py` and `threshold_experiment.py` print exactly the output shown in the article.
- `mutations.py` (Linux, Python 3.15): all 11 deliberate bugs make a test fail.

Not checked in CI: `separation_experiment.py`, which runs a million gradient descent steps.
