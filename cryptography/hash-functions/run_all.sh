#!/bin/sh
set -e
python3 -m unittest discover -s tests -v
python3 digest_demo.py | diff - tests/expected/digest_demo.txt
echo "Python: all checks passed"
