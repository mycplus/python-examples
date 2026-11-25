#!/bin/sh
# Every check the article relies on. Run from this folder: sh run_all.sh
set -e
python3 -m unittest discover -s tests -v
ruff check .
ruff format --check .
mypy knn.py data.py worked_example.py wine_knn.py wine_experiment.py dimensionality_experiment.py benchmark.py mutations.py tests/test_knn.py pitfalls/optimistic_evaluation.py
for script in worked_example wine_knn wine_experiment; do
    python3 "$script.py" | diff --strip-trailing-cr - "tests/expected/$script.txt"
    echo "output matches: $script.py"
done
python3 dimensionality_experiment.py > results/dimensionality.txt
python3 pitfalls/optimistic_evaluation.py > results/optimistic_evaluation.txt
python3 benchmark.py > results/benchmark.txt
python3 mutations.py > results/mutations.txt
echo "all checks passed; experiment outputs written to results/"
