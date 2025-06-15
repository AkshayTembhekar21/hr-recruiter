# HR Recruiter Resume Analyzer Setup Script
Write-Host "`n=== HR Recruiter Resume Analyzer Setup ===" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "`nFound Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "`nError: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists and remove it
if (Test-Path .venv) {
    Write-Host "`nRemoving existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# Create new virtual environment
Write-Host "`nCreating new virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check if credentials.json exists
if (-not (Test-Path credentials.json)) {
    Write-Host "`nWarning: credentials.json not found!" -ForegroundColor Red
    Write-Host "`nPlease follow these steps to get credentials.json:" -ForegroundColor Yellow
    Write-Host "1. Go to https://console.cloud.google.com" -ForegroundColor White
    Write-Host "2. Create a new project" -ForegroundColor White
    Write-Host "3. Enable Gmail API" -ForegroundColor White
    Write-Host "4. Create OAuth 2.0 credentials (Desktop app)" -ForegroundColor White
    Write-Host "5. Download and save as 'credentials.json' in this folder" -ForegroundColor White
} else {
    Write-Host "`nFound credentials.json" -ForegroundColor Green
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "`nTo run the script:" -ForegroundColor Yellow
Write-Host "1. Activate the virtual environment: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Run the script: python gmail_processor.py" -ForegroundColor White

# Keep the window open
Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 