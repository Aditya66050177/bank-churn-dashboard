# Create .python64 directory
$dir = ".python64"
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}

# Download python embeddable 64-bit if not exists
$zipPath = Join-Path $dir "python.zip"
if (-not (Test-Path $zipPath)) {
    Write-Host "Downloading Python 3.10.11 64-bit portable..."
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip" -OutFile $zipPath
}

# Extract if python.exe not exists
$exePath = Join-Path $dir "python.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "Extracting Python..."
    Expand-Archive -Path $zipPath -DestinationPath $dir -Force
}

# Modify python310._pth to enable site-packages
$pthPath = Join-Path $dir "python310._pth"
if (Test-Path $pthPath) {
    Write-Host "Configuring python310._pth..."
    $content = @"
python310.zip
.
Lib\site-packages
import site
"@
    Set-Content -Path $pthPath -Value $content
}

# Download get-pip.py if not exists
$pipScript = Join-Path $dir "get-pip.py"
if (-not (Test-Path $pipScript)) {
    Write-Host "Downloading get-pip.py..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $pipScript
}

# Install pip if not installed
$pipExe = Join-Path $dir "Scripts\pip.exe"
if (-not (Test-Path $pipExe)) {
    Write-Host "Installing pip..."
    Start-Process -FilePath $exePath -ArgumentList $pipScript -Wait -NoNewWindow
}

# Install requirements
Write-Host "Installing requirements..."
Start-Process -FilePath $pipExe -ArgumentList "install -r requirements.txt" -Wait -NoNewWindow

Write-Host "Environment setup completed successfully!"
