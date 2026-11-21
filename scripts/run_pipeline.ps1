# CelebA Face AI - PowerShell Automated Pipeline Runner
param (
    [string]$Stage = "all"
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  CelebA Face AI & Intelligence System - Pipeline Automation Suite" -ForegroundColor Cyan
Write-Host "  Stage Target: $Stage" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "..\workflow\pipeline_runner.py"
python $scriptPath --stage $Stage

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Pipeline completed with zero errors." -ForegroundColor Green
} else {
    Write-Host "`n[FAILURE] Pipeline reported errors." -ForegroundColor Red
}
