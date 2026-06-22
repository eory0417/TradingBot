# TradingBot-Plus PyInstaller one-folder 포터블 빌드
# 사용: powershell -ExecutionPolicy Bypass -File .\build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "오류: .venv 가 없습니다. 먼저 setup.ps1 을 실행하세요." -ForegroundColor Red
    exit 1
}

Write-Host "PyInstaller 설치 확인..."
& $python -m pip install -q pyinstaller

Write-Host "빌드 시작 (수 분~십수 분 소요, 용량 2~4GB)..."
& $python -m PyInstaller --clean -y TradingBot.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dist = Join-Path $PSScriptRoot "dist\TradingBot-Plus"
Copy-Item -Force (Join-Path $PSScriptRoot ".env.example") (Join-Path $dist ".env.example")

@"
@echo off
cd /d "%~dp0"
echo TradingBot-Plus 시작 중... 브라우저에서 http://localhost:8501
TradingBotPlus.exe
pause
"@ | Set-Content -Encoding ASCII (Join-Path $dist "run.bat")

@"
@echo off
cd /d "%~dp0"
echo === Binance public API (인증 불필요) ===
echo [spot ping]
curl -s https://api.binance.com/api/v3/ping
echo.
echo [futures ping]
curl -s https://fapi.binance.com/fapi/v1/ping
echo.
echo.
echo ping 이 {} 이면 공개 API 연결 OK.
echo 잔고 오류가 계속되면 Binance API 키 IP 화이트리스트를 확인하세요.
pause
"@ | Set-Content -Encoding ASCII (Join-Path $dist "check_network.bat")

Write-Host ""
Write-Host "완료: $dist" -ForegroundColor Green
Write-Host "  1) dist\TradingBot-Plus\.env.example 를 .env 로 복사 후 API 키 입력"
Write-Host "  2) TradingBotPlus.exe 또는 run.bat 실행"
Write-Host "  3) 다른 PC로 옮길 때 dist\TradingBot-Plus 폴더 전체를 ZIP"
