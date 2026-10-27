@echo off
title Push Diabetic Retinopathy Project to GitHub
echo =====================================================================
echo  UPLOADING DIABETIC RETINOPATHY PROJECT TO GITHUB
echo  GitHub Account: Rishikumar67
echo  Repository: https://github.com/Rishikumar67/Diabetic-Retinopathy-Screening
echo =====================================================================
echo.
echo Make sure you have created the empty repository on GitHub first:
echo   https://github.com/new  (Name: Diabetic-Retinopathy-Screening)
echo.
pause

echo.
echo Pushing committed code and datasets to GitHub...
"C:\Program Files\Git\cmd\git.exe" push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================================================
    echo  SUCCESS! Your project is now live on GitHub:
    echo  https://github.com/Rishikumar67/Diabetic-Retinopathy-Screening
    echo =====================================================================
) else (
    echo.
    echo [Notice] If you saw 'Repository not found', please ensure:
    echo 1. You opened https://github.com/new in your browser
    echo 2. Created the repository named 'Diabetic-Retinopathy-Screening'
    echo 3. Clicked 'Create repository'
    echo Then run this file again!
)

echo.
pause
