@echo off
title Diabetic Retinopathy Screening System
echo =====================================================================
echo  DIABETIC RETINOPATHY SCREENING SYSTEM USING DEEP LEARNING
echo  Submitted by: Rishi Kumar, Vanshika Pal, Samia Khan
echo  Supervisor: Dr. Gaurav Kumar (2MCA1)
echo =====================================================================
echo.
echo Launching Streamlit Web Application...
echo.

if exist .\.venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe -m streamlit run app.py
) else if exist C:\Users\Rishi\anaconda3\python.exe (
    C:\Users\Rishi\anaconda3\python.exe -m streamlit run app.py
) else (
    python -m streamlit run app.py
)

pause
