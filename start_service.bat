@echo off
echo Starting Query Suggestion Service...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if database exists
if not exist "data\ir_database_combined.db" (
    echo Error: Database file not found: data\ir_database_combined.db
    echo Please ensure the database file exists before starting the service
    pause
    exit /b 1
)

echo Database file found.
echo Starting service on http://localhost:8010
echo Press Ctrl+C to stop the service
echo.

REM Start the service
python run_query_service.py

echo.
echo Service stopped.
pause 