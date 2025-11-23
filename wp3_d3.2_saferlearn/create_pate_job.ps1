#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Creates a PATE job with differential privacy enabled via the API.

.DESCRIPTION
    This script creates a PATE job by:
    1. Fetching active worker UUIDs from the orchestrator
    2. Creating a job request with DP enabled
    3. Submitting the job via PUT request to the API

.PARAMETER DpValue
    Differential privacy parameter value (default: 0.5)

.PARAMETER UseDP
    Enable/disable differential privacy (default: $true)

.PARAMETER NumSamples
    Number of samples to process (default: 1000)

.PARAMETER OrchestratorHost
    Orchestrator host (default: localhost)

.PARAMETER OrchestratorPort
    Orchestrator port (default: 5000)

.EXAMPLE
    .\create_pate_job.ps1 -DpValue 0.5 -UseDP $true
#>

param(
    [float]$DpValue = 0.5,
    [bool]$UseDP = $false,
    [int]$NumSamples = 1000,
    [string]$OrchestratorHost = "localhost",
    [int]$OrchestratorPort = 5000,
    [string]$DataFormat = "mnist",
    [string]$Dataset = "MNIST",
    [int]$NumClasses = 10,
    [string]$Algorithm = "pate",
    [int]$NumTeachers = 30,
    [int]$BatchSize = 5
)

$ErrorActionPreference = "Stop"
$baseUrl = "http://${OrchestratorHost}:${OrchestratorPort}"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Creating PATE Job with DP" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Orchestrator: $baseUrl" -ForegroundColor Yellow
Write-Host "DP Enabled: $UseDP" -ForegroundColor Yellow
Write-Host "DP Value: $DpValue" -ForegroundColor Yellow
Write-Host "Algorithm: $Algorithm" -ForegroundColor Yellow
Write-Host "Number of Teachers: $NumTeachers" -ForegroundColor Yellow
Write-Host "Batch Size: $BatchSize" -ForegroundColor Yellow
Write-Host "Sequential Batching: Enabled" -ForegroundColor Yellow
Write-Host ""

# Step 1: Create job with sequential batching
Write-Host "[1/1] Creating PATE job with sequential batching..." -ForegroundColor Green
Write-Host "      The orchestrator will automatically start/stop teachers in batches" -ForegroundColor Cyan
Write-Host ""

$jobBody = @{
    algorithm = $Algorithm
    datatype = $DataFormat
    dataset = $Dataset
    workers = @()  # Empty - teachers will be started automatically
    nbClasses = $NumClasses
    dpValue = $DpValue
    useDP = $UseDP
    numTeachers = $NumTeachers
    batchSize = $BatchSize
} | ConvertTo-Json -Depth 10

Write-Host "Job Request:" -ForegroundColor Yellow
Write-Host ($jobBody | ConvertFrom-Json | ConvertTo-Json -Depth 10) -ForegroundColor White
Write-Host ""

try {
    $jobResponse = Invoke-WebRequest -Uri "$baseUrl/job" -Method PUT -ContentType "application/json" -Body $jobBody -ErrorAction Stop
    $job = $jobResponse.Content | ConvertFrom-Json

    Write-Host "===========================================" -ForegroundColor Green
    Write-Host "Job Created Successfully!" -ForegroundColor Green
    Write-Host "===========================================" -ForegroundColor Green
    Write-Host "Job UUID: $($job.uuid)" -ForegroundColor Yellow
    Write-Host "Algorithm: $($job.algorithm)" -ForegroundColor Yellow
    Write-Host "State: $($job.state)" -ForegroundColor Yellow
    Write-Host "Number of Teachers: $NumTeachers (in batches of $BatchSize)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The orchestrator is now processing the job:" -ForegroundColor Cyan
    Write-Host "  - Starting teachers in batches of $BatchSize" -ForegroundColor White
    Write-Host "  - Collecting votes from each batch" -ForegroundColor White
    Write-Host "  - Stopping teachers after each batch" -ForegroundColor White
    Write-Host "  - Aggregating all votes at the end" -ForegroundColor White
    Write-Host ""
    Write-Host "Monitor job status:" -ForegroundColor Cyan
    Write-Host "  curl http://${OrchestratorHost}:${OrchestratorPort}/jobs" -ForegroundColor White
    Write-Host "  curl http://${OrchestratorHost}:${OrchestratorPort}/job/$($job.uuid)" -ForegroundColor White
    Write-Host ""

} catch {
    Write-Host "ERROR: Failed to create job" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Red
    }
    exit 1
}

