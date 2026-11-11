@echo off
setlocal EnableExtensions
set "PYTHON_EXE=C:\Users\Administrator\.conda\envs\styleshot-blackwell\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONWARNINGS=ignore::FutureWarning"
set "LOGLEVEL=ERROR"
set "GRADIO_ANALYTICS_ENABLED=False"
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python was not found at:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

pushd "%~dp0"
if "%~1"=="" (
    echo Two file dialogs will open: select CONTENT first, then STYLE.
    echo Default profile: groups A-F, 50 steps, 512x512.
    echo.
)
"%PYTHON_EXE%" "%~dp0profile_style_transfer_stages.py" %*
set "PROFILE_EXIT_CODE=%ERRORLEVEL%"

if not "%PROFILE_EXIT_CODE%"=="0" (
    echo.
    echo Profiling exited with an error. Review comparison_summary.json and the message above.
)
popd
pause
endlocal & exit /b %PROFILE_EXIT_CODE%
