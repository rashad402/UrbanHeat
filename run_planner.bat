@echo off
rem Launch the UrbanHeat Planner and open it in the browser.
rem Double-click this file, or use the Desktop shortcut that points to it.
rem Close the console window to stop the server.

cd /d "%~dp0"
title UrbanHeat Planner (close this window to stop)

rem Already running? Then just open it.
powershell -NoProfile -Command "try { Invoke-WebRequest http://127.0.0.1:8080/api/meta -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    start "" http://127.0.0.1:8080
    exit /b 0
)

rem Open the browser once the server answers (it takes about 15 seconds to load the model).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest http://127.0.0.1:8080/api/meta -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:8080'; break } catch { Start-Sleep -Seconds 1 } }"

echo Starting UrbanHeat Planner on http://127.0.0.1:8080 ...
python -m uvicorn api.planner:app --port 8080

echo.
echo The server stopped. Press any key to close this window.
pause >nul
