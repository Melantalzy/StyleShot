@echo off
setlocal
chcp 65001 >nul
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\styleshot-blackwell\python.exe"
set "PROJECT_DIR=D:\practice_projects\StyleShot-baseline"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "GRADIO_ANALYTICS_ENABLED=False"
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python was not found: %PYTHON_EXE%
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" "%PROJECT_DIR%\styleshot_local_gui.py" --inbrowser

if errorlevel 1 (
    echo.
    echo StyleShot GUI exited with an error. Review the message above.
    pause
)
endlocal
