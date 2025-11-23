#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Kills all PATE job processes including Kafka, orchestrator, and teachers.
    
.DESCRIPTION
    This script forcefully stops:
    1. Kafka containers (Docker)
    2. Orchestrator/API server processes
    3. All teacher/data owner processes
    
.PARAMETER Force
    Force kill processes without confirmation
    
.EXAMPLE
    .\kill_all_pate_processes.ps1
    
.EXAMPLE
    .\kill_all_pate_processes.ps1 -Force
#>

param(
    [switch]$Force = $false
)

$ErrorActionPreference = "Continue"
$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "===========================================" -ForegroundColor Red
Write-Host "Killing All PATE Processes" -ForegroundColor Red
Write-Host "===========================================" -ForegroundColor Red
Write-Host ""

if (-not $Force) {
    $confirm = Read-Host "This will kill all PATE processes. Continue? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Step 1: Stop Kafka containers
Write-Host "[1/3] Stopping Kafka containers..." -ForegroundColor Yellow
$kafkaComposeFile = Join-Path $baseDir "kafka_docker.yml"
if (Test-Path $kafkaComposeFile) {
    docker compose -f $kafkaComposeFile down 2>$null
    Write-Host "Kafka containers stopped" -ForegroundColor Green
} else {
    Write-Host "Kafka compose file not found, skipping..." -ForegroundColor Yellow
}

# Force kill any remaining Kafka containers
$kafkaContainers = docker ps -a --filter "name=kafka" --filter "name=zookeeper" --format "{{.ID}}" 2>$null
if ($kafkaContainers) {
    Write-Host "Force killing remaining Kafka containers..." -ForegroundColor Yellow
    $kafkaContainers | ForEach-Object {
        docker rm -f $_ 2>$null
    }
}

# Step 2: Kill orchestrator/API processes
Write-Host ""
Write-Host "[2/3] Killing orchestrator/API processes..." -ForegroundColor Yellow

# First, try to kill orchestrator using saved PID
$orchestratorPidFile = Join-Path $baseDir "orchestrator_pid.txt"
if (Test-Path $orchestratorPidFile) {
    $savedPid = Get-Content $orchestratorPidFile -ErrorAction SilentlyContinue
    if ($savedPid -and $savedPid -match '^\d+$') {
        Write-Host "Killing orchestrator from orchestrator_pid.txt (PID: $savedPid)..." -ForegroundColor Yellow
        try {
            Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
            Write-Host "Killed orchestrator process: PID $savedPid" -ForegroundColor Green
        } catch {
            # Process may not exist, continue
        }
        Remove-Item $orchestratorPidFile -Force -ErrorAction SilentlyContinue
    }
}

# Also find processes by command line arguments (backup method)
$apiProcesses = Get-CimInstance Win32_Process -Filter "name = 'python.exe' OR name = 'cmd.exe'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*api.py*"
}

if ($apiProcesses) {
    $count = 0
    $apiProcesses | ForEach-Object {
        Write-Host "Killing orchestrator process: PID $($_.ProcessId)" -ForegroundColor Yellow
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $count++
    }
    Write-Host "Killed $count additional orchestrator process(es)" -ForegroundColor Green
} else {
    Write-Host "No additional orchestrator processes found" -ForegroundColor Yellow
}

# Step 3: Kill all teacher/data owner processes
Write-Host ""
Write-Host "[3/3] Killing teacher/data owner processes..." -ForegroundColor Yellow

# First, try to kill processes using saved PIDs
$pidsFile = Join-Path $baseDir "teacher_pids.txt"
if (Test-Path $pidsFile) {
    $savedPids = Get-Content $pidsFile -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\d+$' }
    if ($savedPids) {
        Write-Host "Killing processes from teacher_pids.txt..." -ForegroundColor Yellow
        foreach ($pid in $savedPids) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                Write-Host "  Killed PID: $pid" -ForegroundColor Yellow
            } catch {
                # Process may not exist, ignore
            }
        }
        Remove-Item $pidsFile -Force -ErrorAction SilentlyContinue
    }
}

# Also find processes by command line arguments (backup method)
$teacherProcesses = Get-CimInstance Win32_Process -Filter "name = 'python.exe' OR name = 'cmd.exe'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*data_owner_cli.py*"
}

if ($teacherProcesses) {
    $count = 0
    $teacherProcesses | ForEach-Object {
        Write-Host "Killing teacher process: PID $($_.ProcessId)" -ForegroundColor Yellow
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $count++
    }
    Write-Host "Killed $count additional teacher process(es)" -ForegroundColor Green
} else {
    Write-Host "No additional teacher processes found" -ForegroundColor Yellow
}

# Additional cleanup: Kill any processes on PATE ports (orchestrator and teachers)
Write-Host ""
Write-Host "[Bonus] Checking for processes on PATE ports..." -ForegroundColor Yellow

$portsToCheck = @(5000)  # Orchestrator port
# Add teacher ports (1244-1273 for 30 teachers)
for ($port = 1244; $port -le 1273; $port++) {
    $portsToCheck += $port
}

$killedPorts = @()
foreach ($port in $portsToCheck) {
    $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($connection) {
        $pid = ($connection | Select-Object -First 1).OwningProcess
        if ($pid) {
            Write-Host "Killing process on port $port : PID $pid" -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            $killedPorts += $port
        }
    }
}

if ($killedPorts.Count -gt 0) {
    Write-Host "Killed processes on ports: $($killedPorts -join ', ')" -ForegroundColor Green
} else {
    Write-Host "No processes found on PATE ports" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  - Kafka containers: Stopped" -ForegroundColor White
Write-Host "  - Orchestrator processes: Killed" -ForegroundColor White
Write-Host "  - Teacher processes: Killed" -ForegroundColor White
Write-Host ""
Write-Host "Note: Terminal windows may remain open. Close them manually if needed." -ForegroundColor Yellow
Write-Host ""

