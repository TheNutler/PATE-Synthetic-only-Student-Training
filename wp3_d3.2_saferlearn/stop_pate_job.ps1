#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Stops all PATE job components.
    
.DESCRIPTION
    This script stops:
    1. Kafka (via Docker Compose)
    2. Shows instructions for manually stopping orchestrator and teachers
    
.EXAMPLE
    .\stop_pate_job.ps1
#>

param(
    [switch]$StopKafka = $true,
    [switch]$StopTeachers = $false
)

$ErrorActionPreference = "Stop"
$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Stopping PATE Job Components" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

if ($StopKafka) {
    Write-Host "[1/2] Stopping Kafka..." -ForegroundColor Green
    $kafkaComposeFile = Join-Path $baseDir "kafka_docker.yml"
    if (Test-Path $kafkaComposeFile) {
        docker compose -f $kafkaComposeFile down
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Kafka stopped successfully" -ForegroundColor Green
        } else {
            Write-Host "Warning: Failed to stop Kafka (may already be stopped)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Warning: Kafka compose file not found" -ForegroundColor Yellow
    }
    Write-Host ""
}

if ($StopTeachers) {
    Write-Host "[2/2] Instructions for stopping teachers and orchestrator:" -ForegroundColor Yellow
} else {
    Write-Host "[2/2] Manual cleanup required:" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To stop the orchestrator and teachers:" -ForegroundColor White
Write-Host "  - Press Ctrl+C in each terminal window" -ForegroundColor White
Write-Host "  - Or close the terminal windows manually" -ForegroundColor White
Write-Host ""
Write-Host "Note: Teachers and orchestrator are running in separate terminals." -ForegroundColor Yellow
Write-Host "      Close them manually by pressing Ctrl+C or closing the windows." -ForegroundColor Yellow
Write-Host ""

# Optional: Clean up database
Write-Host "To clean up the database:" -ForegroundColor Cyan
Write-Host "  # Clear workers" -ForegroundColor White
Write-Host "  curl -X DELETE http://localhost:5000/clients" -ForegroundColor White
Write-Host ""
Write-Host "  # Clear jobs" -ForegroundColor White
Write-Host "  curl -X DELETE http://localhost:5000/jobs" -ForegroundColor White
Write-Host ""

