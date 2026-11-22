@echo off
title CelebA Face AI - Google AI Studio Multimodal Application
cd /d "%~dp0\.."
echo ==============================================================================
echo   CelebA Face AI - Google AI Studio Multimodal Application
echo ==============================================================================
echo.
python app/google_ai_studio_app.py %*
pause
