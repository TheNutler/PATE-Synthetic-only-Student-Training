#!/usr/bin/env pwsh
<#
.SYNOPSIS
    View teacher logs.
    
.DESCRIPTION
    View logs from background teacher processes.
    
.PARAMETER TeacherNum
    View log for specific teacher number (1-30). If not specified, shows recent logs from all teachers.
    
.PARAMETER Lines
    Number of lines to show (default: 50)
    
.EXAMPLE
    .\view_teacher_logs.ps1 -TeacherNum 1
    .\view_teacher_logs.ps1 -Lines 100
#>

param(
    [int]$TeacherNum = 0,
    [int]$Lines = 50,
    [switch]$Orchestrator = $false
)

$ErrorActionPreference = "Continue"
$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $baseDir "teacher_logs"

if (-not (Test-Path $logsDir)) {
    Write-Host "Logs directory not found: $logsDir" -ForegroundColor Red
    Write-Host "Teachers may not have been started yet, or logs haven't been created." -ForegroundColor Yellow
    exit 1
}

# View orchestrator log if requested
if ($Orchestrator) {
    $orchestratorLogFile = Join-Path $logsDir "orchestrator_api.log"
    if (Test-Path $orchestratorLogFile) {
        Write-Host "===========================================" -ForegroundColor Cyan
        Write-Host "Orchestrator/API Log" -ForegroundColor Cyan
        Write-Host "===========================================" -ForegroundColor Cyan
        Write-Host ""
        Get-Content $orchestratorLogFile -Tail $Lines -ErrorAction SilentlyContinue
        Write-Host ""
        Write-Host "To follow the log (live updates):" -ForegroundColor Cyan
        Write-Host "  Get-Content $orchestratorLogFile -Tail 50 -Wait" -ForegroundColor White
    } else {
        Write-Host "Orchestrator log file not found: $orchestratorLogFile" -ForegroundColor Red
    }
    exit 0
}

if ($TeacherNum -gt 0) {
    # View specific teacher log
    $logFiles = Get-ChildItem -Path $logsDir -Filter "teacher_${TeacherNum}_*.log" | Sort-Object LastWriteTime -Descending
    if ($logFiles.Count -eq 0) {
        Write-Host "No log file found for Teacher $TeacherNum" -ForegroundColor Red
        exit 1
    }
    $logFile = $logFiles[0]
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host "Teacher $TeacherNum Log: $($logFile.Name)" -ForegroundColor Cyan
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host ""
    Get-Content $logFile.FullName -Tail $Lines -ErrorAction SilentlyContinue
} else {
    # Show recent activity from all teachers
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host "Recent Teacher Logs (Last $Lines lines per teacher)" -ForegroundColor Cyan
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $logFiles = Get-ChildItem -Path $logsDir -Filter "teacher_*.log" | Sort-Object LastWriteTime -Descending
    if ($logFiles.Count -eq 0) {
        Write-Host "No log files found in $logsDir" -ForegroundColor Red
        exit 1
    }
    
    foreach ($logFile in $logFiles) {
        $teacherMatch = $logFile.Name -match "teacher_(\d+)_"
        if ($teacherMatch) {
            $tnum = $matches[1]
            Write-Host "--- Teacher $tnum ($($logFile.Name)) ---" -ForegroundColor Yellow
            Get-Content $logFile.FullName -Tail 10 -ErrorAction SilentlyContinue
            Write-Host ""
        }
    }
    
    Write-Host "To view full log for a specific teacher:" -ForegroundColor Cyan
    Write-Host "  .\view_teacher_logs.ps1 -TeacherNum <number>" -ForegroundColor White
    Write-Host ""
    Write-Host "To view orchestrator log:" -ForegroundColor Cyan
    Write-Host "  .\view_teacher_logs.ps1 -Orchestrator" -ForegroundColor White
}

