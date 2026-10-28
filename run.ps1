Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
& ".\.venv\Scripts\python.exe" main.py
