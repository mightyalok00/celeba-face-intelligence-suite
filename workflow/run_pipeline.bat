@echo off
title CelebA Face AI - Automated Workflow Runner
cd /d "%~dp0\.."

echo ================================================================
echo   CelebA Face AI & Intelligence System - Workflow Runner
echo ================================================================
echo.
echo Executing automated pipeline verification suite...
echo.

python workflow/pipeline_runner.py --stage all

echo.
echo ================================================================
echo Execution complete. Press any key to exit.
echo ================================================================
pause
