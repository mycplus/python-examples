# Run:  .\run_all.ps1
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { Write-Output "Python: tests FAILED"; exit 1 }
Write-Output "Python: all tests passed"
