# Binary Search Tree in Python

[![binary-tree](https://github.com/mycplus/python-examples/actions/workflows/binary-tree.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/binary-tree.yml)

Companion code for [Binary Trees in C, C++, Java, Python and C#](https://www.mycplus.com/computer-science/data-structures/binary-trees/) on MYCPLUS.

| File | What it is |
| --- | --- |
| `binary_tree.py` | A generic binary search tree with unique keys, and the article's demo; insert, search, remove and every traversal run without recursion |
| `recursive_height.py` | The C version's recursive height on chains of 500 to 10,000 nodes, showing where CPython's recursion limit stops it |
| `tests/` | `unittest` tests and the demo's expected output |

## Run and test

```sh
python3 binary_tree.py
python3 -m unittest discover -s tests -v
```

## What the build checks

- `ruff check` and `mypy --strict` pass, on Python 3.10 to 3.14, on Linux and Windows.
- The demo tree's four traversal orders, height and size, and every removal case.
- An empty tree and a `str` tree.
- 200,000 random inserts, removes and membership tests agree with a Python `set`.
- A 3,000-node chain (sorted insertion, three times the default recursion limit) is walked in every order without recursion.
- `binary_tree.py` prints exactly the output shown in the article.
- `recursive_height.py` succeeds at 500 nodes and raises `RecursionError` at 1,000.
