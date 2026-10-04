# Insertion Sort in Python

[![insertion-sort](https://github.com/mycplus/python-examples/actions/workflows/insertion-sort.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/insertion-sort.yml)

Companion code for [Insertion Sort in C, C++, Java, Python and C#](https://www.mycplus.com/computer-science/algorithms/insertion-sort/) on MYCPLUS.

| File | What it is |
| --- | --- |
| `insertion_sort.py` | The article's listing: `insertion_sort()` and a demo that sorts the article's array |
| `tests/test_insertion_sort.py` | `unittest` tests against `sorted()`, and the demo's expected output |

Requires Python 3.10 or later. No third-party packages are needed to run the code; CI installs pinned versions of `ruff` (0.16.9) and `mypy` (2.3.1) for checking.

## Run and test

```sh
python3 insertion_sort.py
python3 -m unittest discover -s tests -v
```

## What the build checks

- Runs on Python 3.10 to 3.14, on Linux and Windows.
- `ruff check` and `mypy --strict` report nothing.
- `insertion_sort()` agrees with `sorted()` on 5,000 random lists of 0 to 64 values (every third one with only 4 distinct values), 20,000 random ints, ascending and descending input, and empty and one-element lists, and sorts in place and returns the same list.
- The demo prints exactly the output shown in the article.
