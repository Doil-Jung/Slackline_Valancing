@echo off
cd /d "%~dp0"

:: Auto-detect: desktop(user) vs laptop(doilm)
if "%USERNAME%"=="user" (
    set PYTHON=C:\Users\user\miniconda3\envs\pybullet\python.exe
) else if "%USERNAME%"=="doilm" (
    set PYTHON=C:\Users\doilm\AppData\Local\Programs\Python\Python311\python.exe
) else (
    echo [ERROR] Unknown user: %USERNAME%
    pause
    exit /b 1
)

"%PYTHON%" analyze_log.py
echo exitcode=%errorlevel%
type compare_out.txt
pause
