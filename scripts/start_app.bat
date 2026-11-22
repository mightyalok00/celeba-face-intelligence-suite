@echo off
title CelebA Face AI - Application Launcher
cd /d "%~dp0\.."
echo ==============================================================================
echo   CelebA Face AI - Application Launcher
echo   1. Google AI Studio Multimodal Face Intelligence Application (Recommended)
echo   2. Streamlit Web Dashboard
echo ==============================================================================
set /p choice="Select application to launch [1 or 2, default: 1]: "
if "%choice%"=="2" (
    echo Starting Streamlit Dashboard (app/app.py)...
    streamlit run app/app.py
) else (
    echo Starting Google AI Studio Application (app/google_ai_studio_app.py)...
    python app/google_ai_studio_app.py
)
pause
