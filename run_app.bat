@echo off
title Titanic Survival Predictor
echo ========================================================
echo        Starting Titanic Survival Prediction App
echo ========================================================
echo.
echo Opening web application at http://127.0.0.1:5000 ...
echo Press CTRL+C in this terminal window to stop the server.
echo.

python app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start.
    echo Please make sure Python and the required packages are installed.
    pause
)

