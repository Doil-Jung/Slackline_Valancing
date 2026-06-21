@echo off
echo Starting V10 PyBullet Verification...

:: Auto-detect: desktop(user) vs laptop(doilm)
if "%USERNAME%"=="user" (
    set PYTHON=C:\Users\user\miniconda3\envs\pybullet\python.exe
    set PATH=C:\Users\user\miniconda3\envs\pybullet;C:\Users\user\miniconda3\envs\pybullet\Library\bin;%PATH%
) else if "%USERNAME%"=="doilm" (
    set PYTHON=C:\Users\doilm\AppData\Local\Programs\Python\Python311\python.exe
) else (
    echo [ERROR] Unknown user: %USERNAME%
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0v10_pybullet.py"
if errorlevel 1 pause
