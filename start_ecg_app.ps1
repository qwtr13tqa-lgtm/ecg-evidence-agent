# start_ecg_app.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$token = Read-Host "请输入网关Token" -AsSecureString
$env:ECG_API_KEY = [System.Net.NetworkCredential]::new("", $token).Password
Remove-Variable token

# 固定网关地址、模型
$env:ECG_BASE_URL = "http://aigw.dlut.edu.cn/v1"
$env:ECG_MODEL = "DeepSeek-V4-Flash-0731-W8A8"

$env:LANGSMITH_TRACING = "false"
$env:LANGCHAIN_TRACING_V2 = "false"

.\.venv-ecg\Scripts\python.exe -m streamlit run app_ecg.py --server.address 127.0.0.1 --server.port 8505 --browser.gatherUsageStats false
