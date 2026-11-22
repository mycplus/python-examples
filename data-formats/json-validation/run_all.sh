#!/bin/sh
# Runs the tests in a virtual environment with the pinned jsonschema, then
# checks that the tests catch three deliberate mistakes.
set -u
cd "$(dirname "$0")"
status=0
report() { printf '%-58s %s\n' "$1" "$2"; }

python3 -m venv .venv >/dev/null && .venv/bin/pip install -q -r requirements.txt >pip.log 2>&1 ||
    { report "virtual environment" FAIL; exit 1; }
py=.venv/bin/python
if $py -m unittest -q test_outputs >test.log 2>&1; then report "unittest ($($py --version))" PASS; else report "unittest" FAIL; status=1; fi

mutant() { # label, file, sed expression
    cp "$2" "$2.orig"
    sed -i "$3" "$2"
    if cmp -s "$2" "$2.orig"; then report "$1" "NOT APPLIED"; status=1
    elif $py -m unittest -q test_outputs >/dev/null 2>&1; then report "$1" MISSED; status=1
    else report "$1" CAUGHT; fi
    mv "$2.orig" "$2"
}
mutant "Mutant 1: NaN and Infinity accepted (default json.loads)" validate_json.py \
    's/json.loads(text, parse_constant=reject_constant)/json.loads(text)/'
mutant "Mutant 2: control characters allowed (strict=False)" validate_json.py \
    's/json.loads(text, parse_constant=reject_constant)/json.loads(text, parse_constant=reject_constant, strict=False)/'
mutant "Mutant 3: only the first schema error reported" validate_user.py \
    's/validator.iter_errors(data)/list(validator.iter_errors(data))[:1]/'
exit $status
