#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Starts the orchestrator/API server for PATE jobs.
    
.DESCRIPTION
    This script starts:
    1. Kafka (via Docker Compose)
    2. Orchestrator/API server (running in foreground - logs appear in terminal)
    
    Teachers are started automatically by the orchestrator in sequential batches
    when you create a PATE job via the API.
    
.PARAMETER KafkaPort
    Kafka port (default: 29092)
    
.PARAMETER OrchestratorHost
    Orchestrator host (default: 127.0.0.1)
    
.PARAMETER OrchestratorPort
    Orchestrator port (default: 5000)
    
.EXAMPLE
    .\start_pate_job.ps1
#>

param(
    [int]$KafkaPort = 29092,
    [string]$OrchestratorHost = "127.0.0.1",
    [int]$OrchestratorPort = 5000
)

$ErrorActionPreference = "Stop"
$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Create logs directory for orchestrator output
$logsDir = Join-Path $baseDir "teacher_logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "Created logs directory: $logsDir" -ForegroundColor Green
}

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Starting PATE Orchestrator/API Server" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Kafka Port: $KafkaPort" -ForegroundColor Yellow
Write-Host "Orchestrator: ${OrchestratorHost}:${OrchestratorPort}" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: Teachers will be started automatically in sequential batches" -ForegroundColor Cyan
Write-Host "      when you create a PATE job via the API." -ForegroundColor Cyan
Write-Host ""

# Step 1: Start Kafka
Write-Host "[1/2] Starting Kafka..." -ForegroundColor Green
$kafkaComposeFile = Join-Path $baseDir "kafka_docker.yml"
if (-not (Test-Path $kafkaComposeFile)) {
    Write-Host "ERROR: Kafka compose file not found: $kafkaComposeFile" -ForegroundColor Red
    exit 1
}

# Check if Kafka is already running
$kafkaRunning = docker ps --filter "name=kafka" --format "{{.Names}}" 2>$null
if ($kafkaRunning -match "kafka") {
    Write-Host "Kafka is already running. Skipping..." -ForegroundColor Yellow
} else {
    Write-Host "Starting Kafka containers..." -ForegroundColor Cyan
    docker compose -f $kafkaComposeFile up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to start Kafka" -ForegroundColor Red
        exit 1
    }
    Start-Sleep -Seconds 5
    Write-Host "Kafka started successfully" -ForegroundColor Green
}

# Step 2: Start Orchestrator/API (in foreground - logs will appear in terminal)
Write-Host ""
Write-Host "[2/2] Starting Orchestrator/API..." -ForegroundColor Green
Write-Host ""

# Display header in terminal
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "ORCHESTRATOR/API SERVER" -ForegroundColor Cyan
Write-Host "Port: $OrchestratorPort" -ForegroundColor Yellow
Write-Host "Kafka: localhost:$KafkaPort" -ForegroundColor Yellow
Write-Host "API: http://${OrchestratorHost}:${OrchestratorPort}" -ForegroundColor Yellow
Write-Host "Started: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Use Python directly from venv
$pythonExe = Join-Path $baseDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

# Set environment variables
$env:KAFKA_HOST = "localhost"
$env:KAFKA_PORT = $KafkaPort

# Change to base directory
Set-Location $baseDir

Write-Host "Orchestrator is running. Logs will appear below:" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the orchestrator" -ForegroundColor Yellow
Write-Host ""

# Start orchestrator process and capture PID
# Output will appear in terminal (not redirected)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $pythonExe
$psi.Arguments = "src/api.py --datasets-path src/input-data/"
$psi.WorkingDirectory = $baseDir
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false
$psi.EnvironmentVariables["KAFKA_HOST"] = "localhost"
$psi.EnvironmentVariables["KAFKA_PORT"] = $KafkaPort

$orchestratorProcess = New-Object System.Diagnostics.Process
$orchestratorProcess.StartInfo = $psi

# Start the process
try {
    $orchestratorProcess.Start() | Out-Null
    
    # Save orchestrator PID for cleanup
    $orchestratorPid = $orchestratorProcess.Id
    $orchestratorPid | Out-File -FilePath (Join-Path $baseDir "orchestrator_pid.txt") -Encoding ASCII
    
    # Wait for process to exit (runs in foreground - output will appear in terminal)
    $orchestratorProcess.WaitForExit()
    
    # If process exited with error, report it
    if ($orchestratorProcess.ExitCode -ne 0) {
        Write-Host "ERROR: Orchestrator exited with code $($orchestratorProcess.ExitCode)" -ForegroundColor Red
        exit $orchestratorProcess.ExitCode
    }
} catch {
    Write-Host "ERROR: Failed to start orchestrator - $_" -ForegroundColor Red
    exit 1
} finally {
    # Clean up PID file when process exits
    $pidFile = Join-Path $baseDir "orchestrator_pid.txt"
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}
