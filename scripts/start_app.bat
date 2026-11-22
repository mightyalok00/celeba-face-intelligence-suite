@echo off
title CelebA Face AI - Google AI Studio Application Launcher
cd /d "%~dp0\.."
echo ==============================================================================
echo   CelebA Face AI - Google AI Studio Multimodal Application
echo ==============================================================================
echo.
python app.py %*
pause
