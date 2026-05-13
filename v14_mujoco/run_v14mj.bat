@echo off
echo Starting V14 Launcher...

:: Auto-detect: desktop(user) vs laptop(doilm)
if "%USERNAME%"=="user" (
    set PYTHON=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe
) else if "%USERNAME%"=="doilm" (
    set PYTHON=C:\Users\doilm\AppData\Local\Programs\Python\Python311\python.exe
) else (
    echo [ERROR] Unknown user: %USERNAME%
    pause
    exit /b 1
)

echo Using: %PYTHON%
"%PYTHON%" -u "%~dp0launcher_v14.py"
if errorlevel 1 pause
