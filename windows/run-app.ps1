$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HELMET_RIDER_CONFIG = "configs/windows-rtx.yaml"
& ".venv\Scripts\python.exe" -m uvicorn app.api:app --host 127.0.0.1 --port 8013
if ($LASTEXITCODE -ne 0) { throw "App stopped with an error." }
