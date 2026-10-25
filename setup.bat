@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo [HandTrack] uv tidak ditemukan.
    echo Install uv dulu:  winget install --id=astral-sh.uv -e
    echo atau lihat https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

echo [HandTrack] Menyiapkan Python 3.12 + dependensi...
uv python install 3.12
if errorlevel 1 goto :error
uv sync
if errorlevel 1 goto :error

echo.
echo [HandTrack] Setup selesai. Jalankan: run.bat
echo.
pause
exit /b 0

:error
echo.
echo [HandTrack] Setup gagal. Periksa pesan di atas.
pause
exit /b 1
