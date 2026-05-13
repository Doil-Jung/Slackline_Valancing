@echo off
chcp 65001 >nul
REM ============================================================
REM  모든 프로젝트 GitHub 자동 동기화 스크립트
REM  5분마다 Windows 작업 스케줄러로 실행됩니다.
REM  변경사항이 있는 프로젝트만 커밋 + 푸시합니다.
REM ============================================================

set BASE=C:\Users\user\OneDrive - 충북과학고등학교\원드라이브 동기화 폴더\코딩 작업 폴더

REM === 프로젝트 목록 (여기에 추가하면 됨) ===
call :sync_project "%BASE%\slackline_valancing"
call :sync_project "%BASE%\acoustic_simulation"
call :sync_project "%BASE%\ai_grader"

exit /b 0

:sync_project
cd /d %1
if not exist ".git" exit /b 0

REM 원격 변경사항 가져오기
git pull --rebase origin main >nul 2>&1

REM 변경사항 확인
git status --porcelain > "%TEMP%\git_status_check.tmp"
set /p STATUS=<"%TEMP%\git_status_check.tmp"
del "%TEMP%\git_status_check.tmp"

if "%STATUS%"=="" exit /b 0

REM 변경사항 있으면 커밋 + 푸시
git add -A
git commit -m "자동 동기화: %date% %time%"
git push origin main >nul 2>&1
exit /b 0
