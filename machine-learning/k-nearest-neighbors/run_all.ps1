# Windows check for the MYCPLUS k-nearest neighbors article.
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
    # stderr lines arrive as error records; keep only their text (an empty
    # line would otherwise print as "System.Management.Automation.RemoteException")
    $text = & $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" }
    } | Out-String
    Add $text.TrimEnd()
    Add '```'
    Add "exit code: $LASTEXITCODE"
    Add ""
}

Set-Location $PSScriptRoot
Add "# Windows results, k-nearest neighbors"
Add ""
Run "Environment" @($py, "-c", "import sys, platform, numpy, sklearn, scipy; print(platform.platform()); print(sys.version); print('numpy', numpy.__version__, 'scikit-learn', sklearn.__version__, 'scipy', scipy.__version__)")
Run "Unit tests" @($py, "-m", "unittest", "discover", "-s", "tests", "-v")
Run "ruff check" @($py, "-m", "ruff", "check", ".")
Run "ruff format --check" @($py, "-m", "ruff", "format", "--check", ".")
Run "mypy" @($py, "-m", "mypy", "knn.py", "data.py", "worked_example.py", "wine_knn.py", "wine_experiment.py", "dimensionality_experiment.py", "benchmark.py", "mutations.py", "tests/test_knn.py", "pitfalls/optimistic_evaluation.py")

foreach ($s in @("worked_example", "wine_knn", "wine_experiment")) {
    $name = Split-Path $s -Leaf
    $actual = (& $py "$s.py" 2>&1 | Out-String).Replace("`r`n", "`n").TrimEnd()
    $expected = (Get-Content -Raw -Encoding UTF8 "tests/expected/$name.txt").Replace("`r`n", "`n").TrimEnd()
    Add "## Output of $s.py"
    if ($actual -eq $expected) { Add "matches tests/expected/$name.txt" }
    else { Add "DIFFERS from tests/expected/$name.txt:"; Add '```text'; Add $actual; Add '```' }
    Add ""
}
Run "dimensionality_experiment.py" @($py, "dimensionality_experiment.py")
Run "pitfalls/optimistic_evaluation.py" @($py, "pitfalls/optimistic_evaluation.py")
Run "benchmark.py (about 30 seconds)" @($py, "benchmark.py")
Run "mutations.py (about 1 to 2 minutes)" @($py, "mutations.py")

[System.IO.File]::WriteAllLines($out, $lines, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Wrote $out"
