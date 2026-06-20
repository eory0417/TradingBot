"""핵심 트레이딩 루프 오케스트레이션 (4단계).

뉴스 분석(2단계) + 기술적 지표/주문 엔진(3단계) + 동적 익절/손절 전략(4단계)을
하나의 비동기 루프로 묶어 실행하고, 결과를 공유 상태(:mod:`state`)에 기록하여
Streamlit GUI가 실시간으로 표시할 수 있게 한다.

동작 모드
---------
  * LIVE 모드: 실제 바이낸스 자격증명이 있으면 ccxt로 시세/주문을 처리한다.
  * SIM 모드 : 자격증명이 플레이스홀더이면 합성 가격(랜덤워크)으로 페이퍼
    트레이딩을 수행한다. 단, 뉴스 파이프라인(무료 RSS + FinBERT)은 두 모드
    모두에서 실제로 동작하므로 GUI를 그대로 시연할 수 있다.

진입 규칙(뉴스 트레이딩)
------------------------
강한 감성 뉴스(|score| >= 임계값)가 특정 코인을 언급하고, 지표가 방향을
확인(긍정+기울기>0 → Long, 부정+기울기<0 → Short)하면 가변형 시장 지정가로
진입한다. 청산은 :class:`strategy.Position`의 동적 익절/손절 규칙을 따른다.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import settings
from logger import get_logger, log_exception
from news_analyzer import AnalyzedNews, NewsAnalyzer
from state import STATE, NewsView, PositionView
from strategy import Position, Side
from trading_engine import TradingEngine, compute_indicators_from_df
from translator import translate_to_korean

log = get_logger(__name__)

# 코인 심볼 <-> 뉴스 키워드 매핑(뉴스 제목에서 대상 코인 탐지).
SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BTC/USDT": ("bitcoin", "btc"),
    "ETH/USDT": ("ethereum", "ether", "eth"),
    "SOL/USDT": ("solana", "sol"),
    "XRP/USDT": ("ripple", "xrp"),
}

# SIM 모드 기준 시작 가격.
SIM_BASE_PRICES: dict[str, float] = {
    "BTC/USDT": 65000.0,
    "ETH/USDT": 3500.0,
    "SOL/USDT": 150.0,
    "XRP/USDT": 0.60,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def detect_symbols(title: str, universe: list[str]) -> list[str]:
    """뉴스 제목에서 언급된 대상 코인 심볼을 추출한다."""
    text = title.lower()
    hits = []
    for symbol in universe:
        for kw in SYMBOL_KEYWORDS.get(symbol, ()):  # 키워드 매칭
            if kw in text:
                hits.append(symbol)
                break
    return hits


def has_real_credentials() -> bool:
    """자격증명이 실제처럼 보이는지(플레이스홀더가 아닌지) 판정."""
    key = settings.binance_api_key.get_secret_value()
    return bool(key) and "your_" not in key


def exchange_mode_label() -> str:
    """현재 거래소 연결 모드 라벨(SIM / DEMO / LIVE)."""
    if not has_real_credentials():
        return "SIM(페이퍼)"
    if settings.binance_testnet:
        return "DEMO"
    return "LIVE"


# --------------------------------------------------------------------------- #
#  SIM 모드용 합성 시장(페이퍼 트레이딩)
# --------------------------------------------------------------------------- #
class SimMarket:
    """랜덤워크 기반 합성 OHLCV/잔고/체결을 제공하는 모의 시장."""

    def __init__(self, symbols: list[str]) -> None:
        self.balance = 10_000.0
        self._closes: dict[str, deque] = {}
        rng = np.random.default_rng(42)
        for sym in symbols:
            base = SIM_BASE_PRICES.get(sym, 100.0)
            # 초기 120개 캔들을 약한 추세 + 노이즈로 시드.
            drift = rng.normal(0, base * 0.0008, 120).cumsum()
            series = base + drift + rng.normal(0, base * 0.0005, 120)
            self._closes[sym] = deque(series.tolist(), maxlen=400)
        self._rng = rng

    def tick(self, symbol: str) -> float:
        """가격을 한 스텝 진행시키고 새 종가를 반환한다."""
        closes = self._closes[symbol]
        last = closes[-1]
        nxt = max(last * (1 + self._rng.normal(0, 0.0015)), 1e-6)
        closes.append(nxt)
        return nxt

    def price(self, symbol: str) -> float:
        return self._closes[symbol][-1]

    def ohlcv(self, symbol: str, limit: int = 120) -> list[list[float]]:
        closes = list(self._closes[symbol])[-limit:]
        now_ms = int(_now().timestamp() * 1000)
        rows = []
        for i, c in enumerate(closes):
            jitter = abs(c) * 0.001
            ts = now_ms - (len(closes) - i) * 60_000
            rows.append([ts, c - jitter, c + jitter, c - jitter, c, 1.0])
        return rows


# --------------------------------------------------------------------------- #
#  트레이딩 봇
# --------------------------------------------------------------------------- #
class TradingBot:
    """뉴스 + 지표 + 전략을 묶는 핵심 오케스트레이터."""

    def __init__(self, state=STATE) -> None:
        self.state = state
        self.symbols = settings.symbols
        self.sim = not has_real_credentials()
        self.notional = settings.position_size_usdt
        self.leverage = settings.leverage
        self.threshold = settings.news_score_threshold
        self.monitor_interval = settings.monitor_interval

        self.positions: dict[str, Position] = {}
        # 심볼별 최신 뉴스 컨텍스트(점수/내용).
        self._latest_news: dict[str, tuple[str, float]] = defaultdict(lambda: ("", 0.0))
        self._lock = asyncio.Lock()
        self._running = False

        # 모드별 구성.
        self.exchange = None
        self.engine: TradingEngine | None = None
        self.notifier = None
        self.sim_market: SimMarket | None = SimMarket(self.symbols) if self.sim else None
        self.news = NewsAnalyzer()

    # ---- 라이프사이클 ----
    async def run(self) -> None:
        """봇을 시작하고 뉴스 태스크 + 모니터 루프를 병행 실행한다."""
        self._running = True
        mode = exchange_mode_label()
        self.state.set_running(True, status=f"running ({mode})")
        self._emit_log("INFO", "system", f"트레이딩 봇 시작 | 모드={mode} | 심볼={self.symbols}")
        log.info("TradingBot starting | mode=%s | symbols=%s", mode, self.symbols)

        try:
            self._emit_log("INFO", "system", "초기화 중 — 거래소/잔고 연결…")
            await self._setup()
            self._emit_log("INFO", "system", "초기화 완료 — 뉴스·차트 수집 시작 (FinBERT 로딩 중일 수 있음)")
            news_task = asyncio.create_task(self.news.start(self._on_news))
            monitor_task = asyncio.create_task(self._monitor_loop())
            await asyncio.gather(news_task, monitor_task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="bot_run")
            self._emit_log("ERROR", "system", f"봇 루프 오류: {exc}")
        finally:
            await self._teardown()
            self.state.set_running(False, status="stopped")
            self._emit_log("INFO", "system", "봇 종료")

    def stop(self) -> None:
        self._running = False
        self.news.stop()
        self.state.set_running(False, status="stopped")

    async def _setup(self) -> None:
        if self.sim:
            self.state.set_balance(self.sim_market.balance)
            self._emit_log("INFO", "system", "SIM 모드: 합성 시장 + 페이퍼 잔고 10,000 USDT")
            return
        # LIVE 모드: 실제 거래소/알림 구성.
        from exchange import create_exchange, load_markets_safe
        from notifier import TelegramNotifier

        self.exchange = create_exchange()
        await load_markets_safe(self.exchange)
        self.notifier = TelegramNotifier()
        self.engine = TradingEngine(self.exchange, notifier=self.notifier)
        bal, bal_err = await self.engine.fetch_balance_usdt()
        self.state.set_balance(bal)
        if bal_err:
            self._emit_log("ERROR", "system", f"잔고 조회 실패: {bal_err}")
        elif exchange_mode_label() == "DEMO" and bal == 0.0:
            self._emit_log(
                "WARNING", "system",
                "Demo 잔고가 0입니다. demo.binance.com API 키인지 확인하세요.",
            )

    async def _teardown(self) -> None:
        try:
            if self.exchange is not None:
                from exchange import close_exchange
                await close_exchange(self.exchange)
            if self.notifier is not None:
                await self.notifier.close()
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="bot_teardown")

    async def _news_context(self, news: str, score: float | None = None) -> str:
        """로그용 뉴스 요약(영문 원문 + 한글 번역)."""
        ko = await asyncio.to_thread(translate_to_korean, news)
        en = news if len(news) <= 120 else news[:117] + "…"
        ko_show = ko if len(ko) <= 120 else ko[:117] + "…"
        if score is not None:
            return f"뉴스({score:+.2f}) | EN: {en} | 한글: {ko_show}"
        return f"EN: {en} | 한글: {ko_show}"

    # ---- 뉴스 콜백(진입 트리거) ----
    async def _on_news(self, item: AnalyzedNews) -> None:
        title_ko = await asyncio.to_thread(translate_to_korean, item.title)
        self.state.add_news(
            NewsView(
                time=_now().strftime("%H:%M:%S"),
                title=item.title,
                score=item.score,
                label=item.label,
                source=item.item.source,
                title_ko=title_ko,
            )
        )
        symbols = detect_symbols(item.title, self.symbols)
        for sym in symbols:
            self._latest_news[sym] = (item.title, item.score)

        # 강한 감성 뉴스만 진입 평가.
        if abs(item.score) < self.threshold:
            return
        for sym in symbols:
            await self._maybe_enter(sym, item.title, item.score)

    async def _maybe_enter(self, symbol: str, news: str, score: float) -> None:
        async with self._lock:
            if symbol in self.positions:
                return
            if len(self.positions) >= settings.max_positions:
                self._emit_log(
                    "WARNING", "entry",
                    f"진입 보류 {symbol}: 동시 포지션 한도({settings.max_positions}) 도달",
                )
                return

        ind = await self._indicators(symbol)
        if ind is None:
            return

        # 뉴스 방향과 지표(기울기) 방향이 일치할 때만 진입.
        side: Side | None = None
        if score > 0 and ind.slope > 0:
            side = "long"
        elif score < 0 and ind.slope < 0:
            side = "short"
        if side is None:
            ctx = await self._news_context(news, score)
            self._emit_log(
                "INFO", "entry",
                f"진입 스킵 {symbol}: {ctx} · 기울기({ind.slope:+.4f}) 방향 불일치",
            )
            return

        await self._open(symbol, side, ind, news, score)

    # ---- 진입 ----
    async def _open(self, symbol: str, side: Side, ind, news: str, score: float) -> None:
        price = ind.last_price
        if self.sim:
            amount = self.notional / price
            margin = self.notional / self.leverage
            self.sim_market.balance -= margin
            self.state.set_balance(self.sim_market.balance)
            order_price = price
            filled = amount
        else:
            result = await self.engine.enter_position(symbol, side)
            if not result.is_filled:
                ctx = await self._news_context(news, score)
                self._emit_log(
                    "ERROR", "order",
                    f"진입 실패 {symbol} {side}: {result.reason} | {ctx}",
                )
                return
            order_price = result.price
            filled = result.filled_amount

        async with self._lock:
            pos = Position(
                symbol=symbol, side=side, amount=filled, entry_price=order_price,
                atr=ind.atr if ind.atr and not np.isnan(ind.atr) else order_price * 0.01,
                entry_news=news, entry_score=score,
            )
            pos.prev_rsi = ind.rsi
            self.positions[symbol] = pos

        ctx = await self._news_context(news, score)
        self._emit_log(
            "INFO", "entry",
            f"진입 {side.upper()} {symbol} | 진입가={order_price:.4f} 수량={filled:.6f} "
            f"금액={self.notional:.2f}USDT | {ctx}",
        )
        self._sync_position_view(self.positions[symbol])

        # 텔레그램 알림(LIVE 모드, 실제 자격증명 시).
        if self.notifier is not None:
            await self.notifier.send_position_open(
                symbol=symbol, side=side, amount_usdt=self.notional,
                entry_price=order_price, news=news, score=score,
            )

    # ---- 모니터 루프(가격 갱신 + 청산 판정) ----
    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._monitor_once()
            except Exception as exc:  # noqa: BLE001 - 루프 생존
                log_exception(log, exc, context="monitor_loop")
            await asyncio.sleep(self.monitor_interval)

    async def _monitor_once(self) -> None:
        for symbol in self.symbols:
            # 가격 진행(SIM) 및 최신 지표 계산.
            if self.sim:
                self.sim_market.tick(symbol)
            ind = await self._indicators(symbol)
            if ind is None:
                continue
            mark = ind.last_price

            # 차트용 OHLCV/라인 상태 저장.
            ohlcv = await self._ohlcv(symbol)
            if ohlcv:
                self.state.set_ohlcv(symbol, ohlcv)

            pos = self.positions.get(symbol)
            if pos is None:
                continue

            news_title, news_score = self._latest_news[symbol]
            signal = pos.update(
                mark, atr=ind.atr, slope=ind.slope, rsi=ind.rsi, news_score=news_score,
            )
            self.state.push_lines(symbol, pos.stop_loss_price, pos.trailing_stop)
            self._sync_position_view(pos)

            if signal.should_exit:
                await self._close(pos, signal)

        # 잔고 갱신(LIVE).
        if not self.sim and self.engine is not None:
            bal, bal_err = await self.engine.fetch_balance_usdt()
            self.state.set_balance(bal)
            if bal_err:
                self._emit_log("ERROR", "system", f"잔고 조회 실패: {bal_err}")

    # ---- 청산 ----
    async def _close(self, pos: Position, signal) -> None:
        exit_price = pos.mark_price
        pnl_pct = pos.unrealized_pct()
        if self.sim:
            margin = self.notional / self.leverage
            pnl_usdt = margin * self.leverage * (pnl_pct / 100)
            self.sim_market.balance += margin + pnl_usdt
            self.state.set_balance(self.sim_market.balance)
        else:
            result = await self.engine.close_position(
                pos.symbol, pos.side, pos.amount, order_type=signal.order_type
            )
            if result.is_filled and result.price:
                exit_price = result.price

        async with self._lock:
            self.positions.pop(pos.symbol, None)
        self.state.remove_position(pos.symbol)

        ctx = await self._news_context(pos.entry_news, pos.entry_score)
        self._emit_log(
            "INFO", "exit",
            f"청산 {pos.side.upper()} {pos.symbol} | 진입가={pos.entry_price:.4f} "
            f"청산가={exit_price:.4f} 손익={pnl_pct:+.2f}% 사유={signal.exit_type} "
            f"({signal.reason}) | {ctx}",
        )

        if self.notifier is not None:
            await self.notifier.send_position_close(
                symbol=pos.symbol, side=pos.side, amount_usdt=self.notional,
                entry_price=pos.entry_price, exit_price=exit_price, pnl_pct=pnl_pct,
                reason=f"{signal.exit_type}: {signal.reason}",
                news=pos.entry_news, score=pos.entry_score,
            )

    # ---- 헬퍼 ----
    async def _ohlcv(self, symbol: str) -> list[list[float]]:
        if self.sim:
            return self.sim_market.ohlcv(symbol)
        df = await self.engine.fetch_ohlcv_df(symbol)
        return df.values.tolist() if df is not None else []

    async def _indicators(self, symbol: str):
        if self.sim:
            ohlcv = self.sim_market.ohlcv(symbol)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            return compute_indicators_from_df(symbol, settings.timeframe, df)
        return await self.engine.compute_indicators(symbol)

    def _sync_position_view(self, pos: Position) -> None:
        self.state.upsert_position(
            PositionView(
                symbol=pos.symbol, side=pos.side, amount=pos.amount,
                entry_price=pos.entry_price, mark_price=pos.mark_price,
                stop_loss=pos.stop_loss_price, trailing_stop=pos.trailing_stop,
                atr_mult=pos.atr_mult, unrealized_pct=pos.unrealized_pct(),
                entry_news=pos.entry_news, entry_score=pos.entry_score,
                opened_at=pos.opened_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    def _emit_log(self, level: str, category: str, message: str) -> None:
        self.state.log(level, category, message)
        getattr(log, level.lower(), log.info)("[%s] %s", category, message)


async def _main() -> None:
    bot = TradingBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Stopped by user")
