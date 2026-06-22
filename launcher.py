"""PyInstaller 진입점 — Streamlit 대시보드를 기동한다.

개발 환경:  python launcher.py
빌드 후:    dist/TradingBot/TradingBot.exe
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _configure_ssl() -> None:
    """PyInstaller 번들에서 aiohttp/ccxt HTTPS 실패 방지 (curl 은 되는데 Python 만 실패)."""
    try:
        import certifi
    except ImportError:
        return
    ca = certifi.where()
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "certifi" / "cacert.pem"
        if bundled.is_file():
            ca = str(bundled)
    os.environ["SSL_CERT_FILE"] = ca
    os.environ["REQUESTS_CA_BUNDLE"] = ca


ROOT = _app_root()
_configure_ssl()
os.chdir(ROOT)

os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "models" / "hf_cache"))
# transformers/torchvision 미설치 시 Streamlit watcher 오류 스팸 방지(dev·exe 공통).
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
if getattr(sys, "frozen", False):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

app_py = ROOT / "app.py"
if not app_py.is_file() and getattr(sys, "frozen", False):
    bundled = Path(getattr(sys, "_MEIPASS", ROOT)) / "app.py"
    if bundled.is_file():
        app_py = bundled

if not app_py.is_file():
    print(f"오류: app.py를 찾을 수 없습니다. ({app_py})", file=sys.stderr)
    raise SystemExit(1)

from streamlit.web import cli as stcli

sys.argv = [
    "streamlit",
    "run",
    str(app_py),
    "--global.developmentMode=false",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
]
raise SystemExit(stcli.main())
