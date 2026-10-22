@echo off
rem CW transmission-mode GUI (app_cw.py). Run from anywhere: uses the venv next to this file.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app_cw.py %*
pause
