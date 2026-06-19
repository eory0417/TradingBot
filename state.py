"""봇 루프와 Streamlit GUI가 공유하는 스레드 안전 상태 저장소 (4단계).

트레이딩 봇은 자체 이벤트 루프를 가진 백그라운드 스레드에서 실행되고,
Streamlit은 메인 스레드에서 주기적으로 화면을 다시 그린다. 두 곳에서 안전하게
읽고 쓰기 위해 모든 접근을 ``threading.Lock``으로 보호하는 단일 :class:`BotState`
싱글톤을 제공한다.

저장 항목
---------
  * 계정 잔고(USDT)
  * 현재 오픈 포지션(심볼별 진입가/방향/익절·손절 라인 등)
  * 심볼별 최근 OHLCV(캔들 차트용)
  * 실시간 뉴스 피드(내용/감성 점수)
  * 이벤트 로그(진입/청산, 주문 실패 사유 등)
  * 사용자 설정(증거금 모드/레버리지/투자금 등)
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PositionView:
    """GUI 표시용 포지션 스냅샷."""

    symbol: str
    side: str
    amount: float
    entry_price: float
    mark_price: float
    stop_loss: float
    trailing_stop: float
    atr_mult: float
    unrealized_pct: float
    entry_news: str
    entry_score: float
    opened_at: str


@dataclass
class NewsView:
    """GUI 표시용 뉴스 항목."""

    time: str
    title: str
    score: float
    label: str
    source: str


class BotState:
    """스레드 안전 공유 상태(싱글톤)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._balance: float = 0.0
        self._positions: dict[str, PositionView] = {}
        self._ohlcv: dict[str, list[list[float]]] = {}
        # TP/SL 라인 히스토리(차트 오버레이용): {symbol: [(ts, stop, trail), ...]}
        self._lines: dict[str, deque] = {}
        self._news: deque[NewsView] = deque(maxlen=200)
        self._logs: deque[dict] = deque(maxlen=500)
        self._settings: dict[str, Any] = {}
        self._running: bool = False
        self._status: str = "stopped"
        self._last_update: str = _now().isoformat()

    # ---- 잔고 ----
    def set_balance(self, balance: float) -> None:
        with self._lock:
            self._balance = float(balance)
            self._touch()

    def get_balance(self) -> float:
        with self._lock:
            return self._balance

    # ---- 포지션 ----
    def upsert_position(self, pos: PositionView) -> None:
        with self._lock:
            self._positions[pos.symbol] = pos
            self._touch()

    def remove_position(self, symbol: str) -> None:
        with self._lock:
            self._positions.pop(symbol, None)
            self._touch()

    def get_positions(self) -> list[PositionView]:
        with self._lock:
            return list(self._positions.values())

    # ---- OHLCV / 라인 ----
    def set_ohlcv(self, symbol: str, ohlcv: list[list[float]]) -> None:
        with self._lock:
            self._ohlcv[symbol] = ohlcv
            self._touch()

    def get_ohlcv(self, symbol: str) -> list[list[float]]:
        with self._lock:
            return list(self._ohlcv.get(symbol, []))

    def push_lines(self, symbol: str, stop: float, trail: float) -> None:
        with self._lock:
            dq = self._lines.setdefault(symbol, deque(maxlen=500))
            dq.append((_now().isoformat(), stop, trail))

    def get_lines(self, symbol: str) -> list[tuple]:
        with self._lock:
            return list(self._lines.get(symbol, []))

    def symbols_with_data(self) -> list[str]:
        with self._lock:
            return list(self._ohlcv.keys())

    # ---- 뉴스 ----
    def add_news(self, news: NewsView) -> None:
        with self._lock:
            self._news.appendleft(news)
            self._touch()

    def get_news(self, limit: int = 50) -> list[NewsView]:
        with self._lock:
            return list(self._news)[:limit]

    # ---- 로그 ----
    def log(self, level: str, category: str, message: str) -> None:
        with self._lock:
            self._logs.appendleft(
                {
                    "time": _now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": level,
                    "category": category,
                    "message": message,
                }
            )
            self._touch()

    def get_logs(self, limit: int = 200, category: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._logs)
        if category:
            items = [it for it in items if it["category"] == category]
        return items[:limit]

    # ---- 설정 ----
    def update_settings(self, **kwargs: Any) -> None:
        with self._lock:
            self._settings.update(kwargs)
            self._touch()

    def get_settings(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    # ---- 실행 상태 ----
    def set_running(self, running: bool, status: str | None = None) -> None:
        with self._lock:
            self._running = running
            if status:
                self._status = status
            self._touch()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def last_update(self) -> str:
        with self._lock:
            return self._last_update

    def snapshot(self) -> dict:
        """전체 상태의 직렬화 가능한 스냅샷(디버깅/표시용)."""
        with self._lock:
            return {
                "balance": self._balance,
                "positions": [asdict(p) for p in self._positions.values()],
                "news": [asdict(n) for n in list(self._news)[:20]],
                "logs": list(self._logs)[:50],
                "settings": dict(self._settings),
                "status": self._status,
                "running": self._running,
            }

    def _touch(self) -> None:
        self._last_update = _now().isoformat()


# 프로세스 전역 싱글톤. 봇 스레드와 Streamlit이 동일 인스턴스를 공유한다.
STATE = BotState()
