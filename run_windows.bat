@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || goto folder_error

if not exist "demo.py" (
    echo demo.py is missing. Put this launcher next to demo.py.
    goto failed
)

set "VENV_PROBE=import platform, struct, sys; sys.exit(sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8 or platform.machine().upper() not in ('AMD64', 'X86_64') or sys.prefix == sys.base_prefix)"
set "VENV_PY=.venv\Scripts\python.exe"
call :check_venv
if not errorlevel 1 goto check_models

set "YOLO_VENV_DIR=.venv-yolo312"
set "VENV_PY=%YOLO_VENV_DIR%\Scripts\python.exe"
call :check_venv
if not errorlevel 1 goto check_models

set "YOLO_LAUNCHER=%CD%\%~nx0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { $lines=[IO.File]::ReadAllLines($env:YOLO_LAUNCHER); $start=[Array]::IndexOf($lines,':embedded_python_setup'); if ($start -lt 0) { throw 'Python setup section is missing.' }; $body=$lines[($start+1)..($lines.Length-1)] -join [Environment]::NewLine; & ([scriptblock]::Create($body)); exit 0 } catch { [Console]::Error.WriteLine(($_ | Out-String)); exit 1 }"
if errorlevel 1 goto failed

call :check_venv
if errorlevel 1 goto failed

:check_models
call :models_ready
if not errorlevel 1 goto check_runtime

if not exist "requirements-export.txt" (
    echo requirements-export.txt is missing. Extract the full YOLO demo package.
    goto failed
)
if not exist "export_model.py" (
    echo export_model.py is missing. Extract the full YOLO demo package.
    goto failed
)
echo Installing export dependencies and preparing YOLO models...
"%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
"%VENV_PY%" -m pip install -r requirements-export.txt
if errorlevel 1 goto failed
"%VENV_PY%" export_model.py --int8
if errorlevel 1 goto failed
call :models_ready
if errorlevel 1 (
    echo Model export did not produce all FP32, FP16 and INT8 model files.
    goto failed
)

:check_runtime
"%VENV_PY%" -I -c "import cv2, numpy, openvino" >nul 2>&1
if not errorlevel 1 goto launch
if not exist "requirements.txt" (
    echo requirements.txt is missing. Extract the full YOLO demo package.
    goto failed
)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
"%VENV_PY%" -I -c "import cv2, numpy, openvino"
if errorlevel 1 goto failed

:launch
"%VENV_PY%" demo.py %*
if errorlevel 1 goto failed
popd
exit /b 0

:check_venv
if not exist "%VENV_PY%" exit /b 1
"%VENV_PY%" -I -c "%VENV_PROBE%" >nul 2>&1
exit /b %ERRORLEVEL%

:models_ready
for %%P in (fp32 fp16 int8) do (
    if not exist "models\yolo11n-416-%%P.xml" exit /b 1
    if not exist "models\yolo11n-416-%%P.bin" exit /b 1
)
exit /b 0

:failed
echo The demo stopped. Check the error above for details.
popd
pause
exit /b 1

:folder_error
echo Could not open the YOLO demo folder.
pause
exit /b 1

:embedded_python_setup
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -LiteralPath $env:YOLO_LAUNCHER
Set-Location -LiteralPath $projectRoot
$venvPath = Join-Path $projectRoot $env:YOLO_VENV_DIR
$pythonProbe = "import platform, struct, sys; sys.exit(1) if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8 or platform.machine().upper() not in ('AMD64', 'X86_64') else print(sys._base_executable)"

function Get-Python312 {
    param([string]$Command, [string[]]$CommandArgs = @())

    $ErrorActionPreference = 'Continue'
    $result = & $Command @CommandArgs -I -c $pythonProbe 2>$null
    if ($LASTEXITCODE -eq 0 -and $result -is [string] -and
        (Test-Path -LiteralPath $result -PathType Leaf)) {
        return $result
    }
}

function Find-Python312 {
    $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = Get-Python312 -Command $launcher.Source -CommandArgs @('-3.12-64')
        if ($found) { return $found }
    }

    $paths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    $onPath = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -notlike '*\Microsoft\WindowsApps\python.exe') {
        $paths += $onPath.Source
    }
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $found = Get-Python312 -Command $path
            if ($found) { return $found }
        }
    }
}

function Install-Python312 {
    # Try both installers if the first one cannot provide a compatible interpreter.
    $ErrorActionPreference = 'Continue'
    $manager = Get-Command pymanager.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($manager) {
        Write-Host 'Preparing Python 3.12 with Python Install Manager...'
        & $manager.Source install 3.12 | Out-Host
        $found = Get-Python312 -Command $manager.Source -CommandArgs @('exec', '-V:3.12')
        if ($found) { return $found }
    }

    $winget = Get-Command winget.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host 'Installing Python 3.12 x64 for this user with WinGet...'
        & $winget.Source install --id Python.Python.3.12 --exact --source winget --architecture x64 --scope user --silent --accept-package-agreements --accept-source-agreements --disable-interactivity | Out-Host
        $found = Find-Python312
        if ($found) { return $found }
    }

    throw 'Automatic setup could not find or install Python 3.12 x64. Install it from python.org, then run this launcher again.'
}

$python = Find-Python312
if (-not $python) { $python = Install-Python312 }

if (Test-Path -LiteralPath $venvPath) {
    $backupName = $env:YOLO_VENV_DIR + '.backup-' + [guid]::NewGuid().ToString('N')
    Rename-Item -LiteralPath $venvPath -NewName $backupName
    Write-Host "Saved the previous YOLO environment as $backupName."
}

Write-Host "Creating the YOLO environment with $python..."
& $python -I -m venv $venvPath
if ($LASTEXITCODE -ne 0) { throw 'Creating the YOLO environment failed.' }
