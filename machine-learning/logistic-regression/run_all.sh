#!/bin/sh
# Every check the article relies on. Run from this folder: sh run_all.sh
set -e
python3 -m unittest discover -s tests -v
ruff check .
ruff format --check .
mypy logistic_regression.py data.py tests/test_logistic_regression.py worked_example.py breast_cancer_experiment.py threshold_experiment.py separation_experiment.py mutations.py pitfalls/default_regularization.py
for script in worked_example threshold_experiment; do
    python3 "$script.py" | diff --strip-trailing-cr - "tests/expected/$script.txt"
    echo "output matches: $script.py"
done
python3 breast_cancer_experiment.py > results/breast_cancer.txt
python3 pitfalls/default_regularization.py > results/default_regularization.txt
python3 separation_experiment.py > results/separation.txt
python3 mutations.py > results/mutations.txt
echo "all checks passed; experiment outputs written to results/"
