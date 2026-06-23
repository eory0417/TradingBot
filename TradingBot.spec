# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-folder 빌드: pyinstaller --clean TradingBot.spec"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

import certifi

block_cipher = None

LOCAL_MODULES = [
    "app.py",
    "bot.py",
    "config.py",
    "state.py",
    "strategy.py",
    "trading_engine.py",
    "news_analyzer.py",
    "notifier.py",
    "finetune.py",
    "chart_data.py",
    "exchange.py",
    "logger.py",
    "kst_util.py",
    "translator.py",
    "telegram_news.py",
    "telegram_login.py",
    "finbert_paths.py",
    "http_session.py",
]

datas = [
    (".streamlit", ".streamlit"),
    (".env.example", "."),
    (certifi.where(), "certifi"),
]
datas += [(name, ".") for name in LOCAL_MODULES]

hiddenimports = list(collect_submodules("ccxt"))
hiddenimports += list(collect_submodules("feedparser"))
hiddenimports += list(collect_submodules("telethon"))
hiddenimports += [
    name.removesuffix(".py").replace("/", ".")
    for name in LOCAL_MODULES
]

binaries = []
for pkg in (
    "streamlit",
    "torch",
    "transformers",
    "plotly",
    "pandas_ta",
    "numba",
    "llvmlite",
    "pydantic_settings",
    "deep_translator",
    "certifi",
    "telethon",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports = list(dict.fromkeys(h for h in hiddenimports if isinstance(h, str)))

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingBotPlus",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TradingBot-Plus",
)
