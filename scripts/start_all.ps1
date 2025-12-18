# Unified startup script for FastAPI API and Streamlit UI
# Usage: powershell -ExecutionPolicy Bypass -File .\start_all.ps1
# Stops both processes when this window is closed.

$ErrorActionPreference = 'Stop'

Write-Host "Starting FastAPI (port 8000) and Streamlit (port 8501)..." -ForegroundColor Cyan

# Resolve project root (directory of this script)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

# Optional: Activate virtual environment if exists (looking for .venv)
$venvPath = Join-Path $scriptDir '.venv'
if (Test-Path (Join-Path $venvPath 'Scripts\Activate.ps1')) {
    Write-Host "Activating virtual environment at $venvPath" -ForegroundColor Yellow
    . (Join-Path $venvPath 'Scripts\Activate.ps1')
}

# Start FastAPI via uvicorn
$fastApiCmd = "uvicorn src.interface.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload"
$streamlitCmd = "streamlit run src/interface/web/streamlit_app.py --server.port 8501"

$fastApiProc = Start-Process -FilePath powershell -ArgumentList "-NoLogo", "-NoProfile", "-Command", $fastApiCmd -PassThru
$streamlitProc = Start-Process -FilePath powershell -ArgumentList "-NoLogo", "-NoProfile", "-Command", $streamlitCmd -PassThru

Write-Host "FastAPI PID: $($fastApiProc.Id)" -ForegroundColor Green
Write-Host "Streamlit PID: $($streamlitProc.Id)" -ForegroundColor Green

# Health check loop (basic)
Write-Host "Waiting for services to respond..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
try {
    $apiHealthy = (Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200
} catch { $apiHealthy = $false }
if ($apiHealthy) {
    Write-Host "FastAPI health endpoint OK" -ForegroundColor Green
} else {
    Write-Host "FastAPI health endpoint not reachable yet" -ForegroundColor Yellow
}

Write-Host "Open UI: http://localhost:8501" -ForegroundColor Magenta
Write-Host "API Docs (if enabled): http://localhost:8000/docs" -ForegroundColor Magenta

Write-Host "Press Ctrl+C to terminate both services." -ForegroundColor Yellow

# Wait on both; handle Ctrl+C
try {
    while (-not ($fastApiProc.HasExited -and $streamlitProc.HasExited)) {
        Start-Sleep -Seconds 2
        if ($fastApiProc.HasExited) { Write-Host "FastAPI exited." -ForegroundColor Red }
        if ($streamlitProc.HasExited) { Write-Host "Streamlit exited." -ForegroundColor Red }
        if ($fastApiProc.HasExited -and $streamlitProc.HasExited) { break }
    }
} finally {
    Write-Host "Shutting down child processes..." -ForegroundColor Cyan
    foreach ($p in @($fastApiProc, $streamlitProc)) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force } catch {}
        }
    }
    Pop-Location
    Write-Host "All services stopped." -ForegroundColor Green
}
