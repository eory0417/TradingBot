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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kst_util import format_kst


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
    entry_news_ko: str = ""
    leverage: int = 1
    opened_at_ms: int = 0   # 진입 시각(epoch ms) — 차트 진입 표시용
    news_triggered_at_ms: int = 0  # 뉴스 인식(진입 트리거) 시각(epoch ms)
    added: bool = False     # 추가 진입(피라미딩) 1회 수행 여부
    notional: float = 0.0   # 총 명목금액(USDT, 추가 진입 포함)
    trailing_active: bool = False  # 이익 구간 진입 후 Trailing 활성


@dataclass
class NewsView:
    """GUI 표시용 뉴스 항목."""

    time: str
    title: str
    score: float
    label: str
    source: str
    title_ko: str = ""
    at_ms: int = 0          # 봇이 뉴스를 수신·처리한 시각(epoch ms)
    published_at_ms: int = 0  # RSS 발행 시각(epoch ms) — GUI 표시용


class BotState:
    """스레드 안전 공유 상태(싱글톤)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._balance: float = 0.0
        self._positions: dict[str, PositionView] = {}
        self._ohlcv: dict[str, list[list[float]]] = {}
        self._news: deque[NewsView] = deque(maxlen=200)
        self._logs: deque[dict] = deque(maxlen=500)
        # 수동 청산 요청 큐(GUI → 봇 모니터 루프). 심볼 집합.
        self._close_requests: set[str] = set()
        # 청산 완료된 거래 기록(수익률 통계용).
        self._closed_trades: list[dict] = []
        self._settings: dict[str, Any] = {}
        self._running: bool = False
        self._status: str = "stopped"
        # 프로그램(세션) 시작 시각 — 통계 기본 시작점.
        self._session_start: datetime = _now()
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

    def clear_positions(self) -> None:
        """모든 포지션 스냅샷을 비운다(클린 스타트용)."""
        with self._lock:
            self._positions.clear()
            self._touch()

    def get_positions(self) -> list[PositionView]:
        with self._lock:
            return list(self._positions.values())

    # ---- 수동 청산 요청 ----
    def request_close(self, symbol: str) -> None:
        """GUI에서 특정 코인의 청산을 요청한다(봇이 다음 점검 주기에 처리)."""
        with self._lock:
            self._close_requests.add(symbol)
            self._touch()

    def pop_close_requests(self) -> list[str]:
        """누적된 청산 요청을 모두 꺼내 비운다(봇 모니터 루프에서 호출)."""
        with self._lock:
            reqs = list(self._close_requests)
            self._close_requests.clear()
            return reqs

    # ---- 청산 거래 기록(통계) ----
    def record_trade(self, trade: dict) -> None:
        """청산 완료된 거래 한 건을 기록한다."""
        with self._lock:
            self._closed_trades.append(trade)
            self._touch()

    def get_trades(self) -> list[dict]:
        with self._lock:
            return list(self._closed_trades)

    @property
    def session_start(self) -> datetime:
        with self._lock:
            return self._session_start

    # ---- OHLCV / 라인 ----
    def set_ohlcv(self, symbol: str, ohlcv: list[list[float]]) -> None:
        with self._lock:
            self._ohlcv[symbol] = ohlcv
            self._touch()

    def get_ohlcv(self, symbol: str) -> list[list[float]]:
        with self._lock:
            return list(self._ohlcv.get(symbol, []))

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

    def clear_news(self) -> None:
        """뉴스 피드 표시를 비운다(봇 재시작 시 이전 세션 잔여 제거)."""
        with self._lock:
            self._news.clear()
            self._touch()

    # ---- 로그 ----
    def log(self, level: str, category: str, message: str) -> None:
        with self._lock:
            now = _now()
            self._logs.appendleft(
                {
                    "time": format_kst(now),
                    "time_ms": int(now.timestamp() * 1000),
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

    def _touch(self) -> None:
        self._last_update = _now().isoformat()


# 프로세스 전역 싱글톤. 봇 스레드와 Streamlit이 동일 인스턴스를 공유한다.
STATE = BotState()
