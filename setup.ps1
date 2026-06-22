# TradingBot 가상환경 설정 (Python 3.12 권장 — 3.14는 pandas_ta/numba 미지원)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path .venv) {
    Write-Host "기존 .venv 삭제 중..."
    Remove-Item -Recurse -Force .venv
}

Write-Host "Python 3.12 가상환경 생성 중..."
py -3.12 -m venv .venv

Write-Host "패키지 설치 중 (수 분 소요)..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host ".env 생성됨 (.env.example 복사) — SIM 모드로 실행 가능, 실거래 시 키를 입력하세요."
}

Write-Host ""
Write-Host "완료. 아래 명령으로 실행하세요:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m streamlit run app.py"
