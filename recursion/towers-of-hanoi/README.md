# Towers of Hanoi in Python

[![towers-of-hanoi](https://github.com/mycplus/python-examples/actions/workflows/towers-of-hanoi.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/towers-of-hanoi.yml)

Companion code for [Towers of Hanoi: Recursive Algorithm With Code in C, C++, Java, Python and C#](https://www.mycplus.com/computer-science/algorithms/towers-of-hanoi/) on MYCPLUS.

| File | What it is |
| --- | --- |
| `towers_of_hanoi.py` | The article's listing: prints the moves for 3 disks |
| `hanoi.py` | `solve()` and `solve_iterative()` generators yielding `Move` tuples, `move_at()`, `move_count()`, and a command line: `python3 hanoi.py N [--iterative]` |
| `pitfalls/base_case_one.py` | A base case of `n == 1`: `RecursionError` for 0 disks |
| `tests/test_hanoi.py` | `unittest` tests: simulation, solver agreement, `move_at` for 64, 65 and 100 disks against a reference, input validation and the listing's output |

Requires Python 3.10 or later. No third-party packages are needed to run the code; CI installs pinned versions of `ruff` (0.16.9) and `mypy` (2.3.1) for checking.

## Run and test

```sh
python3 towers_of_hanoi.py
python3 hanoi.py 4 --iterative
python3 -m unittest discover -s tests -v
```

## What the build checks

- Runs on Python 3.10 to 3.14, on Linux and Windows.
- `ruff check` and `mypy --strict` report nothing.
- Both solvers produce identical, legal, complete sequences for 0 to 15 disks, and `move_at()` agrees with an independent reference for 64, 65 and 100 disks.
- The command line accepts only ASCII digits from 0 to 20.
- The listing prints exactly the output shown in the article, and the pitfall stops with `RecursionError`.
