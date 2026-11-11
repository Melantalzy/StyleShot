@echo off
setlocal
chcp 65001 >nul
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\styleshot-blackwell\python.exe"
set "PROJECT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "GRADIO_ANALYTICS_ENABLED=False"
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" "%PROJECT_DIR%run_attention_timestep_pair.py" %*

if errorlevel 1 (
    echo.
    echo Experiment exited with an error. Review the message above.
)
pause
endlocal
