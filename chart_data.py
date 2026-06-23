"""차트용 OHLCV 조회 (Streamlit 팝업 등 UI 전용).

공개 시세 API 로 캔들을 가져오므로 API 키 없이도 동작한다.
SIM/LIVE 모두 동일한 바이낸스 USDⓈ-M 선물 시세를 사용한다.
"""

from __future__ import annotations

import ccxt

from config import settings
from logger import get_logger, log_exception

log = get_logger(__name__)

# UI 에서 선택 가능한 타임프레임.
CHART_TIMEFRAMES: tuple[str, ...] = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")

_exchange: ccxt.binance | None = None
_markets_loaded = False


def _get_exchange() -> ccxt.binance:
    """차트 조회용 ccxt 인스턴스를 프로세스당 1회만 생성·재사용한다."""
    global _exchange, _markets_loaded
    if _exchange is None:
        _exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
            }
        )
        if settings.binance_testnet:
            try:
                _exchange.enable_demo_trading(True)
            except Exception:  # noqa: BLE001
                pass
    if not _markets_loaded:
        _exchange.load_markets()
        _markets_loaded = True
    return _exchange


def fetch_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 200) -> list[list[float]]:
    """심볼의 OHLCV 캔들을 동기로 조회한다.

    반환: ``[[ts, open, high, low, close, volume], ...]``
    실패 시 빈 리스트.
    """
    if timeframe not in CHART_TIMEFRAMES:
        timeframe = "15m"
    try:
        rows = _get_exchange().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return rows or []
    except Exception as exc:  # noqa: BLE001
        log_exception(log, exc, context="chart_fetch_ohlcv", symbol=symbol, timeframe=timeframe)
        return []
