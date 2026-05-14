$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"

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

if ($pythonExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($python312)) {
    Write-Host "Python 3.12 was not found."
    Write-Host "Install Python 3.12, then run this script again:"
    Write-Host "  .\setup_env.ps1"
    exit 1
}

Write-Host "Using Python 3.12 at $python312"
& py -3.12 -m venv $venvPath
& (Join-Path $venvPath "Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $venvPath "Scripts\python.exe") -m pip install -r (Join-Path $projectRoot "requirements.txt")
Write-Host "Environment ready. Activate it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
