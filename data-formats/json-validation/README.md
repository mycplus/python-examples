# JSON Validation in Python

[![JSON Validation](https://github.com/mycplus/python-examples/actions/workflows/json-validation.yml/badge.svg)](https://github.com/mycplus/python-examples/actions/workflows/json-validation.yml)

Companion code for [Valid JSON: Rules, Common Errors and How to Validate It](https://www.mycplus.com/programming/how-to-validate-json/) on MYCPLUS.

| File | What it does |
| --- | --- |
| `validate_json.py` | Reports whether each file is valid JSON, rejecting the NaN and Infinity that `json.loads` accepts by default |
| `validate_user.py` | Validates user records against `cases/schema/user.schema.json` with jsonschema (Draft 2020-12) |
| `edge_cases.py` | Shows what `json.loads` does by default with duplicate keys, big integers, a BOM, a lone surrogate, NaN and deep nesting |
| `test_outputs.py` | Compares each program's output with `tests/expected` |
| `cases/` | The test files, shared with the C++ and JavaScript examples |
| `run_all.sh` | Runs the tests in a virtual environment and three deliberate mistakes the tests must catch |

```sh
pip install -r requirements.txt
python -m unittest -v test_outputs
```

The expected messages were recorded with Python 3.13.
