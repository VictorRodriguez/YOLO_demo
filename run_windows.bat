@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1

set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto check_models

echo Creating a local Python environment...
py -3.12 -c "import platform, sys, struct; sys.exit(sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8 or platform.machine().upper() not in ('AMD64', 'X86_64'))" >nul 2>&1
if not errorlevel 1 goto create_with_launcher
python -c "import platform, sys, struct; sys.exit(sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8 or platform.machine().upper() not in ('AMD64', 'X86_64'))" >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv .venv
if errorlevel 1 goto failed
goto check_models

:create_with_launcher
py -3.12 -m venv .venv
if errorlevel 1 goto failed

:check_models
for %%P in (fp32 fp16 int8) do (
    if not exist "models\yolo11n-416-%%P.xml" goto export_models
    if not exist "models\yolo11n-416-%%P.bin" goto export_models
)
"%VENV_PY%" -c "from importlib.util import find_spec; import sys; sys.exit(not all(find_spec(name) for name in ('cv2', 'numpy', 'openvino')))" >nul 2>&1
if not errorlevel 1 goto launch
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
goto launch

:export_models
echo Installing dependencies and preparing YOLO models. First run can take a few minutes.
"%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
"%VENV_PY%" -m pip install -r requirements-export.txt
if errorlevel 1 goto failed
"%VENV_PY%" export_model.py --int8
if errorlevel 1 goto failed

:launch
"%VENV_PY%" demo.py %*
if errorlevel 1 goto failed
popd
exit /b 0

:python_missing
echo Install Python 3.12 for Windows x86-64 from python.org, then run this file again.
goto show_error

:failed
echo The demo stopped. Check the error above for details.

:show_error
popd
pause
exit /b 1
