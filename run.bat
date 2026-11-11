@echo off
rem Start the InstantHMR -> VRM live app. Extra arguments are passed through,
rem e.g.  run.bat --source camera:0  or  run.bat --no-live
setlocal
cd /d "%~dp0"

if exist "%~dp0..\.venv\Scripts\python.exe" (
  set "PY=%~dp0..\.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" -m app.server %*
