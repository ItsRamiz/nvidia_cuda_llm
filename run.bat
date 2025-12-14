@echo off
setlocal enabledelayedexpansion

set CONTAINER_NAME=pgvector-db
set VENV_PATH=.venv
set APP_FILE=app.py

echo =========================================
echo Activating virtual environment
echo =========================================

IF NOT EXIST "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at %VENV_PATH%
    echo.
    pause
    exit /b 1
)

call "%VENV_PATH%\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo ERROR: Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)

python --version

echo =========================================
echo Checking Docker container: %CONTAINER_NAME%
echo =========================================

docker inspect %CONTAINER_NAME% >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Docker container "%CONTAINER_NAME%" does not exist.
    echo.
    pause
    exit /b 1
)

FOR /F "tokens=*" %%i IN ('docker inspect -f "{{.State.Running}}" %CONTAINER_NAME%') DO set RUNNING=%%i

IF "%RUNNING%"=="false" (
    echo Container is stopped. Starting container...
    docker start %CONTAINER_NAME%
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to start Docker container.
        echo.
        pause
        exit /b 1
    )
) ELSE (
    echo Container is already running.
)

echo =========================================
echo Waiting for Postgres to be ready...
echo =========================================

timeout /t 5 >nul

echo =========================================
echo Running application
echo =========================================

python %APP_FILE%
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Python application failed.
    echo.
    pause
    exit /b 1
)

echo =========================================
echo Application finished successfully
echo =========================================

exit /b 0
