# Build the ns-TTR Simulator desktop app into dist\ns-TTR Simulator\.
# First run creates .venv with the pinned runtime packages + build tools.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.14 -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements-build.txt
}

.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean ns_ttr.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Host "`nBuilt: $PSScriptRoot\dist\ns-TTR Simulator\ns-TTR Simulator.exe"
