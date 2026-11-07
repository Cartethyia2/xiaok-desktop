@echo off
cd /d "%~dp0"
set LOG=%TEMP%\xiaok_error_%RANDOM%_%RANDOM%.txt
"%LOCALAPPDATA%\Programs\Python\Python315\python.exe" main.py 2>"%LOG%"
if errorlevel 1 (echo. & type "%LOG%" & pause)
del "%LOG%" 2>nul
