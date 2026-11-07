@echo off
REM Double-click or run from anywhere: installs packages if needed, starts the app, opens the browser.
cd /d "%~dp0"
python -m pip install -q -r requirements-serving.txt
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8000"
python -m uvicorn serving.app:app --port 8000
pause
