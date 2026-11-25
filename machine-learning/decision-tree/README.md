# Decision Trees in Python From Scratch

[![decision-tree](https://github.com/mycplus/python-examples/actions/workflows/decision-tree.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/decision-tree.yml)

Companion code for [Decision Trees: Entropy, Information Gain and Gini, With Python and C++](https://www.mycplus.com/computer-science/ai/decision-tree/) on MYCPLUS. The trained tree is exported to a text file and to C++ source; the C++ version in [cpp-examples/machine-learning/decision-tree](https://github.com/mycplus/cpp-examples/tree/main/machine-learning/decision-tree) loads both and must print the same predictions.

| File | What it is |
| --- | --- |
| `tree.py` | `DecisionTreeClassifier`: CART with Gini or entropy, `max_depth`, `min_samples_split`, `min_samples_leaf`, NumPy only |
| `data.py` | The breast cancer split (the same as the logistic regression article), the test CSV for C++, and the benchmark row order |
| `worked_example.py` | Eight customers and one feature: Gini and entropy choose different first splits |
| `breast_cancer_experiment.py` | Depth and criterion chosen by repeated cross-validation, test results, scaling, and the chosen tree printed as rules (about 30 seconds) |
| `stability_experiment.py` | scikit-learn's tree with 20 random seeds, and 200 trees on bootstrap samples with their majority vote |
| `export.py`, `export_model.py` | The tree as an outline, as a node table (`models/tree.txt`) and as C++ nested ifs (`models/breast_cancer_tree.hpp`) |
| `predict_exported.py` | Loads the node table with plain Python and prints the report the C++ program must match |
| `benchmark.py` | Prediction time per row: plain Python, this NumPy code and scikit-learn |
| `mutations.py` | Re-applies 17 plausible bugs to a copy of the code and checks the tests catch each (about 1 minute) |
| `tests/` | `unittest` tests and the expected output of three programs |
| `results/` | Outputs captured for the article, plus the data behind its figures |

## Data

The [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) data (569 tumours, 30 features) ships with scikit-learn as `load_breast_cancer`. Wolberg, Mangasarian, Street and Street (1993), UCI Machine Learning Repository, [doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), licensed CC BY 4.0. Labels are flipped so that 1 = malignant. Nothing is downloaded.

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
- 19 tests: impurity values, the worked example, every split against a brute-force search, tie rules, thresholds and boundary values, depth and size limits, identical trees to scikit-learn wherever no two splits tie, scaling, the exported files and generated code, the shared benchmark rows and checksum, and input validation.
- `worked_example.py`, `predict_exported.py` and `breast_cancer_experiment.py` print exactly the output shown in the article.
- `mutations.py` (Linux, Python 3.15): all 17 deliberate bugs make a test fail.

Not checked in CI: `benchmark.py`, whose timings depend on the machine.
