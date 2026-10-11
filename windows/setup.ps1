param()
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Run-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program" }
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Install Python 3.12 (64-bit), then reopen PowerShell. See README.md."
}
Run-Checked "py" @("-3.12", "-m", "venv", ".venv")
$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
Run-Checked $Python @("-m", "pip", "install", "--upgrade", "pip")
Run-Checked $Python @("-m", "pip", "install", "torch==2.9.1", "torchvision==0.24.1", "--index-url", "https://download.pytorch.org/whl/cu128")
Run-Checked $Python @("-m", "pip", "install", "-r", "requirements-windows.txt")
Run-Checked $Python @("training/check_cuda.py")
Write-Host "Setup finished. Download/extract the transfer bundle before training."
