#!/bin/sh
# Every check the article relies on. Run from this folder: sh run_all.sh
set -e
python3 -m unittest discover -s tests -v
ruff check .
ruff format --check .
mypy linear_regression.py tests/test_linear_regression.py worked_example.py \
    diabetes_experiment.py scaling_experiment.py mutations.py pitfalls/*.py
for script in worked_example pitfalls/missing_intercept pitfalls/coefficient_scale; do
    python3 "$script.py" | diff --strip-trailing-cr - "tests/expected/$(basename "$script").txt"
    echo "output matches: $script.py"
done
python3 diabetes_experiment.py > results/diabetes.txt
python3 scaling_experiment.py > results/scaling.txt
python3 pitfalls/normal_equation_collinear.py > results/normal_equation_collinear.txt
python3 mutations.py > results/mutations.txt
echo "all checks passed; experiment outputs written to results/"
