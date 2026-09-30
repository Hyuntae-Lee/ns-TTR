@echo off
rem Rebuild the ns-TTR Simulator desktop app (double-click or run from cmd).
rem Output: dist\ns-TTR Simulator\ns-TTR Simulator.exe
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"
if errorlevel 1 (
    echo.
    echo Build FAILED.
) else (
    echo.
    echo Build finished.
)
pause
