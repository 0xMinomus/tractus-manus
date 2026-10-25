@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" collect_hand_data.py %*
    exit /b %errorlevel%
)

echo.
echo [HandTrack] Virtual environment belum ada.
echo Jalankan setup.bat terlebih dahulu.
echo.
pause
exit /b 1
