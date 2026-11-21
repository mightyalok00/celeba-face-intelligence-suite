@echo off
title CelebA Face AI - Streamlit Dashboard
cd /d "%~dp0\.."
echo Starting CelebA Face AI Dashboard (app/app.py)...
streamlit run app/app.py
pause
