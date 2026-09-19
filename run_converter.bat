@echo off
title VR3D Studio - 2D to 3D SBS & VR180 Converter
cd /d "%~dp0"
echo ========================================================
echo   VR3D Studio - AI 2D to 3D SBS and VR180 Converter
echo   NVIDIA RTX 5070 CUDA Gyorsitassal
echo ========================================================
echo.
echo Inditas...
python main.py
if errorlevel 1 (
    echo.
    echo Hiba tortent az alkalmazas futasa kozben.
    pause
)
