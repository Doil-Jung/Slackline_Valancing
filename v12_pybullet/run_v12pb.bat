@echo off
echo Starting V12 PyBullet Simulation...

:: Auto-detect: desktop(user) vs laptop(doilm)
if "%USERNAME%"=="user" (
    set PYTHON=C:\Users\user\miniconda3\envs\pybullet\python.exe
) else if "%USERNAME%"=="doilm" (
    set PYTHON=C:\Users\doilm\AppData\Local\Programs\Python\Python311\python.exe
) else (
    echo [ERROR] Unknown user: %USERNAME%
    echo Please add your Python path to this batch file.
    pause
    exit /b 1
)

echo Using: %PYTHON%
"%PYTHON%" -u "%~dp0v12_pybullet.py"
if errorlevel 1 pause
