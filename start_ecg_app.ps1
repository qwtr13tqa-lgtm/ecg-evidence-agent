param([switch]$LocalOnly)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv-ecg\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Missing .venv-ecg; see docs/SETUP.md" }
if (-not $LocalOnly) {
    if ([string]::IsNullOrWhiteSpace($env:ECG_BASE_URL)) { $env:ECG_BASE_URL = Read-Host "Gateway Base URL" }
    if ([string]::IsNullOrWhiteSpace($env:ECG_MODEL)) { $env:ECG_MODEL = Read-Host "Model name" }
    if ([string]::IsNullOrWhiteSpace($env:ECG_API_KEY)) {
        $token = Read-Host "Gateway API key" -AsSecureString
        try { $env:ECG_API_KEY = [System.Net.NetworkCredential]::new("", $token).Password }
        finally { Remove-Variable token }
    }
    & $python -c "from src.agent.public_config import gateway_config; gateway_config(); print('Gateway configuration format OK; connectivity not tested.')"
    if ($LASTEXITCODE -ne 0) { throw "Invalid gateway configuration" }
} else {
    Remove-Item Env:ECG_API_KEY -ErrorAction SilentlyContinue
}
$env:LANGSMITH_TRACING = "false"
$env:LANGCHAIN_TRACING_V2 = "false"
& $python -m streamlit run app_ecg.py --server.address 127.0.0.1 --server.port 8505 --browser.gatherUsageStats false
