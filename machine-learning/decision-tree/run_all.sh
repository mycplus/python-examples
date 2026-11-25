#!/bin/sh
# Every check the article relies on. Run from this folder: sh run_all.sh
set -e
python3 -m unittest discover -s tests -v
ruff check .
ruff format --check .
mypy tree.py data.py export.py export_model.py predict_exported.py worked_example.py breast_cancer_experiment.py stability_experiment.py benchmark.py mutations.py tests/test_tree.py
for script in worked_example predict_exported breast_cancer_experiment; do
    python3 "$script.py" | diff --strip-trailing-cr - "tests/expected/$script.txt"
    echo "output matches: $script.py"
done
python3 stability_experiment.py > results/stability.txt
python3 benchmark.py > results/benchmark.txt
python3 mutations.py > results/mutations.txt
echo "all checks passed; experiment outputs written to results/"
