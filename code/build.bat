@echo off
setlocal

REM === Initialize MSVC environment (required by nvcc) ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

IF ERRORLEVEL 1 (
    echo Failed to initialize MSVC environment
    exit /b 1
)

REM === CUDA build ===
set "CU=vector_add.cu"
set "EXE=vector_add.exe"

nvcc "%CU%" -o "%EXE%"

IF ERRORLEVEL 1 (
    echo Build FAILED
    exit /b 1
)

echo Build SUCCESS: %EXE%
endlocal
