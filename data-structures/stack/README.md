# Stack in Python

[![stack](https://github.com/mycplus/python-examples/actions/workflows/stack.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/stack.yml)

Companion code for [Stack Implementation in C, C++, Java, Python and C#](https://www.mycplus.com/computer-science/data-structures/stack-implementation/) on MYCPLUS.

| File | What it is |
| --- | --- |
| `stack.py` | `ArrayStack` (list-backed), `LinkedStack`, a bracket checker and the article's demo |
| `list_growth.py` | Counts list reallocations over 1,000,000 appends |
| `tests/test_stack.py` | `unittest` tests, and the demo's expected output |

Requires Python 3.10 or later. No third-party packages are needed to run the code; CI installs pinned versions of `ruff` (0.16.9) and `mypy` (2.3.1) for checking.

## Run and test

```sh
python3 stack.py
python3 -m unittest discover -s tests -v
```

## What the build checks

- Runs on Python 3.10 to 3.14, on Linux and Windows.
- `ruff check` and `mypy --strict` report nothing.
- Both stacks agree with a plain list over 100,000 random operations.
- `pop()` and `peek()` on an empty stack raise `StackUnderflowError`, a subclass of `IndexError`.
- `None` is stored as an ordinary value.
- A `LinkedStack` of 1,000,000 nodes is deleted without hitting a recursion limit.
- The demo prints exactly the output shown in the article.

`list_growth.py` is not checked: its count depends on the CPython version.
