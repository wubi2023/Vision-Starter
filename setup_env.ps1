$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$localPython = Join-Path $projectRoot ".python\tools\python.exe"
$venvName = ".venv"
$pythonCommand = $null
$pythonArgs = @()
$pythonLabel = $null

if (Test-Path $localPython) {
    $pythonCommand = $localPython
    $venvName = ".venv312"
    $pythonLabel = & $pythonCommand --version
} else {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $python312 = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        $pythonExitCode = $LASTEXITCODE
    } catch {
        $python312 = $null
        $pythonExitCode = 1
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($pythonExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($python312)) {
        $pythonCommand = "py"
        $pythonArgs = @("-3.12")
        $pythonLabel = "Python 3.12 at $python312"
    } else {
        $currentPython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $currentPython) {
            Write-Host "No usable Python was found."
            Write-Host "Install Python 3.12, then run this script again:"
            Write-Host "  .\setup_env.ps1"
            exit 1
        }

        $pythonCommand = $currentPython.Source
        $pythonVersion = & $pythonCommand --version
        $pythonLabel = "$pythonVersion at $pythonCommand"
        Write-Host "Python 3.12 was not found. Falling back to $pythonLabel"
        Write-Host "If dependency installation fails, install Python 3.12 and rerun this script."
    }
}

$venvPath = Join-Path $projectRoot $venvName
Write-Host "Creating local virtual environment at $venvPath"
& $pythonCommand @pythonArgs -m venv $venvPath

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Failed to create local virtual environment at $venvPath"
    exit 1
}

Write-Host "Installing packages into the project-local environment."
& $venvPython -m pip install --disable-pip-version-check --no-cache-dir -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Environment ready. Activate it with:"
Write-Host "  .\$venvName\Scripts\Activate.ps1"
