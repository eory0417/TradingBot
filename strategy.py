"""동적 익절/손절 전략 (4단계) — Long/Short 완벽 대칭 구현.

세 가지 청산 규칙을 캡슐화한 :class:`Position`을 제공한다.

  1. 고정 손절(Fixed Stop)
     진입가 대비 ``stop_loss_pct``% 만큼 불리하게 움직이면 즉시 **시장가** 청산.

  2. 동적 익절(Trailing Stop)
     진입 시 ``ATR * 3.0`` 거리에 익절 라인을 세팅하고, 가격이 유리하게
     움직이면 라인을 따라 올린다(Long)/내린다(Short). 아래 가중치 축소 조건이
     충족되면 ATR 배수를 ``1.5``로 줄여 익절 라인을 현재가에 바짝 붙인다.

       * Long  축소: 기울기 우상향(slope>0) + 긍정 뉴스(score>0.7)  또는
                     RSI가 50을 강하게 상향 돌파.
       * Short 축소: 기울기 우하향(slope<0) + 부정 뉴스(score<-0.7) 또는
                     RSI가 50을 강하게 하향 돌파.

  3. 시간 청산(Time Exit)
     뚜렷한 추세 없이(가중치 축소가 발동되지 않은 상태) 진입 후
     ``time_exit_hours`` 시간이 지나면 **시장 지정가**로 전량 청산.

규칙 우선순위는 손절 → 트레일링 → 시간 청산 순이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from config import settings

Side = Literal["long", "short"]

# RSI 50 강한 돌파 판정 마진(상향: >55, 하향: <45).
RSI_MID = 50.0
RSI_STRONG_MARGIN = 5.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ExitSignal:
    """청산 신호."""

    should_exit: bool
    reason: str = ""
    exit_type: str = ""          # 'stop_loss' | 'trailing_stop' | 'time_exit'
    order_type: str = "market"   # 'market' | 'marketable_limit'


def strong_rsi_cross_up(prev_rsi: Optional[float], rsi: float) -> bool:
    """RSI가 50을 강하게 상향 돌파했는지 여부."""
    if prev_rsi is None:
        return False
    return prev_rsi < RSI_MID and rsi >= RSI_MID + RSI_STRONG_MARGIN


def strong_rsi_cross_down(prev_rsi: Optional[float], rsi: float) -> bool:
    """RSI가 50을 강하게 하향 돌파했는지 여부."""
    if prev_rsi is None:
        return False
    return prev_rsi > RSI_MID and rsi <= RSI_MID - RSI_STRONG_MARGIN


def should_tighten(
    side: Side,
    *,
    slope: Optional[float],
    news_score: Optional[float],
    prev_rsi: Optional[float],
    rsi: Optional[float],
    news_threshold: float,
) -> bool:
    """익절 라인 가중치 축소(ATR 배수 1.5) 조건 충족 여부를 판정한다."""
    if side == "long":
        trend_news = (slope is not None and slope > 0) and (
            news_score is not None and news_score > news_threshold
        )
        rsi_break = rsi is not None and strong_rsi_cross_up(prev_rsi, rsi)
        return bool(trend_news or rsi_break)
    else:  # short
        trend_news = (slope is not None and slope < 0) and (
            news_score is not None and news_score < -news_threshold
        )
        rsi_break = rsi is not None and strong_rsi_cross_down(prev_rsi, rsi)
        return bool(trend_news or rsi_break)


@dataclass(slots=True)
class Position:
    """동적 익절/손절 상태를 가진 오픈 포지션."""

    symbol: str
    side: Side
    amount: float
    entry_price: float
    atr: float                       # 진입 시점(및 최신) ATR
    # 진입 시점 컨텍스트(텔레그램/GUI 표시용).
    entry_news: str = ""
    entry_score: float = 0.0
    order_id: Optional[str] = None
    opened_at: datetime = field(default_factory=_now)

    # 설정 스냅샷(진입 시점 고정).
    stop_loss_pct: float = field(default_factory=lambda: settings.stop_loss_pct)
    atr_mult_base: float = field(default_factory=lambda: settings.trailing_atr_mult)
    atr_mult_tight: float = field(default_factory=lambda: settings.trailing_atr_mult_tight)
    news_threshold: float = field(default_factory=lambda: settings.news_score_threshold)
    time_exit_hours: float = field(default_factory=lambda: settings.time_exit_hours)

    # 동적 상태(초기화 시 계산).
    stop_loss_price: float = 0.0
    trailing_stop: float = 0.0
    atr_mult: float = 0.0
    tightened: bool = False
    highest_price: float = 0.0       # Long 트레일링용 최고가
    lowest_price: float = 0.0        # Short 트레일링용 최저가
    prev_rsi: Optional[float] = None
    mark_price: float = 0.0

    def __post_init__(self) -> None:
        self.atr_mult = self.atr_mult_base
        self.mark_price = self.entry_price
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        if self.side == "long":
            self.stop_loss_price = self.entry_price * (1 - self.stop_loss_pct / 100)
            self.trailing_stop = self.entry_price - self.atr * self.atr_mult
        else:
            self.stop_loss_price = self.entry_price * (1 + self.stop_loss_pct / 100)
            self.trailing_stop = self.entry_price + self.atr * self.atr_mult

    # ---- 손익 ----
    def unrealized_pct(self) -> float:
        """현재가 기준 미실현 손익률(%)."""
        if not self.entry_price:
            return 0.0
        change = (self.mark_price - self.entry_price) / self.entry_price * 100
        return round(change if self.side == "long" else -change, 3)

    # ---- 핵심: 갱신 및 청산 판정 ----
    def update(
        self,
        mark_price: float,
        *,
        atr: Optional[float] = None,
        slope: Optional[float] = None,
        rsi: Optional[float] = None,
        news_score: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> ExitSignal:
        """현재가/지표/뉴스로 트레일링 라인을 갱신하고 청산 여부를 판정한다.

        ``atr``/``slope``/``rsi``/``news_score``는 최신 지표가 있을 때만 전달하면
        되며(예: 15분봉 갱신 시), 없으면 직전 값을 유지한다.
        """
        now = now or _now()
        self.mark_price = mark_price
        if atr is not None and atr > 0:
            self.atr = atr

        # ---- 1) 트레일링 가중치 축소 판정(한 번 축소되면 유지) ----
        if not self.tightened and should_tighten(
            self.side,
            slope=slope,
            news_score=news_score,
            prev_rsi=self.prev_rsi,
            rsi=rsi,
            news_threshold=self.news_threshold,
        ):
            self.tightened = True
            self.atr_mult = self.atr_mult_tight

        if rsi is not None:
            self.prev_rsi = rsi

        # ---- 2) 극값 갱신 및 트레일링 라인 이동(유리한 방향으로만 래칫) ----
        distance = self.atr * self.atr_mult
        if self.side == "long":
            self.highest_price = max(self.highest_price, mark_price)
            candidate = self.highest_price - distance
            self.trailing_stop = max(self.trailing_stop, candidate)
        else:
            self.lowest_price = min(self.lowest_price, mark_price)
            candidate = self.lowest_price + distance
            self.trailing_stop = min(self.trailing_stop, candidate)

        # ---- 3) 고정 손절(시장가) ----
        if self.side == "long" and mark_price <= self.stop_loss_price:
            return ExitSignal(
                True,
                reason=f"fixed stop-loss hit ({self.stop_loss_pct}%): "
                f"{mark_price:.4f} <= {self.stop_loss_price:.4f}",
                exit_type="stop_loss",
                order_type="market",
            )
        if self.side == "short" and mark_price >= self.stop_loss_price:
            return ExitSignal(
                True,
                reason=f"fixed stop-loss hit ({self.stop_loss_pct}%): "
                f"{mark_price:.4f} >= {self.stop_loss_price:.4f}",
                exit_type="stop_loss",
                order_type="market",
            )

        # ---- 4) 동적 익절(트레일링 스톱, 시장가) ----
        if self.side == "long" and mark_price <= self.trailing_stop:
            return ExitSignal(
                True,
                reason=f"trailing-stop hit (ATR x{self.atr_mult}): "
                f"{mark_price:.4f} <= {self.trailing_stop:.4f}",
                exit_type="trailing_stop",
                order_type="market",
            )
        if self.side == "short" and mark_price >= self.trailing_stop:
            return ExitSignal(
                True,
                reason=f"trailing-stop hit (ATR x{self.atr_mult}): "
                f"{mark_price:.4f} >= {self.trailing_stop:.4f}",
                exit_type="trailing_stop",
                order_type="market",
            )

        # ---- 5) 시간 청산(횡보 시, 시장 지정가) ----
        # 가중치 축소(강한 추세)가 발동되지 않은 '추세 없는 횡보' 상태에서만 적용.
        held = now - self.opened_at
        if not self.tightened and held >= timedelta(hours=self.time_exit_hours):
            hours = held.total_seconds() / 3600
            return ExitSignal(
                True,
                reason=f"time exit: held {hours:.1f}h >= {self.time_exit_hours}h "
                f"without a clear trend (sideways)",
                exit_type="time_exit",
                order_type="marketable_limit",
            )

        return ExitSignal(False)
