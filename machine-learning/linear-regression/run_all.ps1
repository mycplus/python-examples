# Windows check for the MYCPLUS linear regression article.
# Run from this folder in PowerShell:  .\run_all.ps1
# Writes results-windows.md next to this script; send that file back.
$ErrorActionPreference = "Continue"
$out = Join-Path $PSScriptRoot "results-windows.md"
$py = "python"
$lines = New-Object System.Collections.Generic.List[string]
function Add($s) { $script:lines.Add($s) }
function Run($title, [string[]]$cmd) {
    Add "## $title"
    Add '```text'
    # "$_" turns stderr records into plain text (no NativeCommandError noise)
    $text = & $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1 | ForEach-Object { "$_" } | Out-String
    Add $text.TrimEnd()
    Add '```'
    Add "exit code: $LASTEXITCODE"
    Add ""
}

Set-Location $PSScriptRoot
Add "# Windows results, linear regression"
Add ""
Run "Environment" @($py, "-c", "import sys, platform, numpy, sklearn, scipy; print(platform.platform()); print(sys.version); print('numpy', numpy.__version__, 'scikit-learn', sklearn.__version__, 'scipy', scipy.__version__)")
Run "Unit tests" @($py, "-m", "unittest", "discover", "-s", "tests", "-v")
Run "ruff check" @($py, "-m", "ruff", "check", ".")
Run "ruff format --check" @($py, "-m", "ruff", "format", "--check", ".")
Run "mypy" @($py, "-m", "mypy", "linear_regression.py", "tests/test_linear_regression.py", "worked_example.py", "diabetes_experiment.py", "scaling_experiment.py", "mutations.py", "pitfalls/missing_intercept.py", "pitfalls/coefficient_scale.py", "pitfalls/normal_equation_collinear.py")

foreach ($s in @("worked_example", "pitfalls/missing_intercept", "pitfalls/coefficient_scale")) {
    $name = Split-Path $s -Leaf
    $actual = (& $py "$s.py" 2>&1 | Out-String).Replace("`r`n", "`n").TrimEnd()
    $expected = (Get-Content -Raw -Encoding UTF8 "tests/expected/$name.txt").Replace("`r`n", "`n").TrimEnd()
    Add "## Output of $s.py"
    if ($actual -eq $expected) { Add "matches tests/expected/$name.txt" }
    else { Add "DIFFERS from tests/expected/$name.txt:"; Add '```text'; Add $actual; Add '```' }
    Add ""
}
Run "diabetes_experiment.py" @($py, "diabetes_experiment.py")
Run "scaling_experiment.py (about 15 to 60 seconds)" @($py, "scaling_experiment.py")
Run "pitfalls/normal_equation_collinear.py" @($py, "pitfalls/normal_equation_collinear.py")
Run "mutations.py (about 1 minute)" @($py, "mutations.py")

[System.IO.File]::WriteAllLines($out, $lines, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Wrote $out"
