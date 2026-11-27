# How We Test Machine Learning Code

Companion code for [How We Test Machine Learning Code: Splits, Baselines, Ranges](https://www.mycplus.com/computer-science/ai/how-we-test-machine-learning-code/) on MYCPLUS. One dataset, one model and one baseline, run 50 times so the page can show how far a reported score moves when only the split changes, and how much tighter a paired comparison is.

| File | What it is |
| --- | --- |
| `protocol_check.py` | 50 stratified 75/25 splits of the breast cancer data, a scaled logistic regression and a majority-class baseline, the paired difference with its standard error, and a from-scratch least-squares fit checked against scikit-learn on the same split |
| `results/protocol_check.txt` | The Windows capture the article quotes |
| `results/protocol_check-ubuntu.txt` | The Ubuntu capture; it differs from the Windows one in exactly two lines, the two agreement gaps |
| `results/split_scores.json` | Every per-split score, the summaries and the agreement gaps, for the Windows run |
| `results/split_scores-ubuntu.json` | The same for the Ubuntu run |

## Data

Breast Cancer Wisconsin (Diagnostic) ships with scikit-learn as `load_breast_cancer` (569 rows, 30 features, 212 malignant and 357 benign). Distributed inside scikit-learn under its BSD-3-Clause licence; nothing is downloaded.

## Run

```sh
python -m pip install -r requirements.txt
python protocol_check.py
```

The script writes `split_scores.json` into the directory you pass as the first argument, or next to itself by default, and prints the numbers quoted on the page. It is deterministic: the same 50 seeds every run.

## What the two captures show

Both systems used Python with NumPy 2.5.5 and scikit-learn 1.9.1: Windows 11 with Python 3.14.6 and Ubuntu 24.04.5 under WSL 2 with Python 3.12.3. All 50 scores and every summary statistic matched to the last printed digit. The from-scratch least-squares fit agreed with scikit-learn to 4.330e-15 on predictions on Windows and 2.554e-15 on Ubuntu, both far inside the 1e-10 relative tolerance the series uses. The difference between the systems is floating-point reassociation, which is why the check is written as a tolerance rather than as an equality.

The script is a demonstration rather than a library, so it has no test suite: it asserts nothing and reports everything. The algorithm articles it supports carry their own test suites.
