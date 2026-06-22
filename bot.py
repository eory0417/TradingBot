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

import finetune
from config import settings
from logger import format_exception_brief, get_logger, log_exception
from news_analyzer import AnalyzedNews, NewsAnalyzer
from state import STATE, NewsView, PositionView
from strategy import ExitSignal, Position, Side
from kst_util import format_kst
from trading_engine import TradingEngine, compute_indicators_from_df
from translator import translate_to_korean

log = get_logger(__name__)


def score_to_leverage(score: float) -> int:
    """뉴스 점수 절대값 → 레버리지 배율(0.7→1x, 0.8→2x, 0.9→3x, 1.0→4x)."""
    a = abs(score)
    if a >= 0.9995:
        return 4
    if a >= 0.9:
        return 3
    if a >= 0.8:
        return 2
    return 1


def auto_leverage_decision(score: float, slope: float) -> tuple[Side | None, int]:
    """뉴스 점수 강도에 따른 (방향, 레버리지) 결정(자동 레버리지 모드).

    반환값 ``(side, leverage)``. ``side`` 가 ``None`` 이면 진입하지 않는다.

    절대값 기준 레버리지: 0.7~ → 1배, 0.8~ → 2배, 0.9~ → 3배, 1.0 → 4배.
    추세 일치(점수>0 & 기울기>0, 점수<0 & 기울기<0)면 위 배율로 진입한다.
    추세가 역방향이라도 ``|점수| == 1.0``(최대 확신)이면 추세 필터를 무시하고
    **2배**로 진입한다. 그 외 역방향은 진입하지 않는다.
    """
    a = abs(score)
    if a < settings.news_score_threshold:
        return None, 0
    extreme = a >= 0.9995  # 사실상 1.0
    base = score_to_leverage(score)

    if score > 0:
        if slope > 0:
            return "long", base
        return ("long", 2) if extreme else (None, 0)
    if score < 0:
        if slope < 0:
            return "short", base
        return ("short", 2) if extreme else (None, 0)
    return None, 0

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
        # 진입 파라미터(명목금액/레버리지/임계값/모니터 주기)는 캐시하지 않고
        # 사용 시점에 settings 에서 직접 읽어 '실시간 설정 반영'을 지원한다.
        # 단, 이미 진입한 포지션은 진입 시 스냅샷한 값을 그대로 유지한다.

        # 수동 재학습 트리거(GUI 버튼 → _finetune_loop 가 감지).
        self._finetune_now = False

        self.positions: dict[str, Position] = {}
        # 심볼별 최신 뉴스 컨텍스트(점수/내용).
        self._latest_news: dict[str, tuple[str, float]] = defaultdict(lambda: ("", 0.0))
        self._lock = asyncio.Lock()
        self._running = False
        self._started_at: datetime | None = None

        # 모드별 구성.
        self.exchange = None
        self.engine: TradingEngine | None = None
        self.notifier = None
        self.sim_market: SimMarket | None = SimMarket(self.symbols) if self.sim else None
        self.news = NewsAnalyzer()

    # ---- 라이프사이클 ----
    async def run(self) -> None:
        """봇을 시작하고 뉴스 태스크 + 모니터 루프를 병행 실행한다."""
        self._started_at = _now()
        self.state.clear_news()
        self._running = True
        mode = exchange_mode_label()
        self.state.set_running(True, status=f"running ({mode})")
        self._emit_log("INFO", "system", f"트레이딩 봇 시작 | 모드={mode} | 심볼={self.symbols}")
        log.info("TradingBot starting | mode=%s | symbols=%s", mode, self.symbols)

        try:
            self._emit_log("INFO", "system", "초기화 중 — 거래소/잔고 연결…")
            await self._setup()
            self._emit_log("INFO", "system", "초기화 완료 — 뉴스·차트 수집 시작 (FinBERT 로딩 중일 수 있음)")
            self._emit_log(
                "INFO", "system",
                f"뉴스 워밍업: 최근 기사는 화면 표시 · 진입은 시작 후 "
                f"{settings.news_entry_grace_seconds}초 + 발행 "
                f"{settings.news_max_age_minutes:.0f}분 이내만",
            )
            news_task = asyncio.create_task(
                self.news.start(
                    self._on_news,
                    on_status=lambda msg: self._emit_log("INFO", "system", msg),
                )
            )
            monitor_task = asyncio.create_task(self._monitor_loop())
            finetune_task = asyncio.create_task(self._finetune_loop())
            await asyncio.gather(news_task, monitor_task, finetune_task)
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
        self.positions.clear()
        self.state.clear_positions()
        self.state.pop_close_requests()
        self.state.set_running(False, status="stopped")

    async def _setup(self) -> None:
        if self.sim:
            # 클린 스타트: 이전 세션의 잔여 포지션 표시를 비우고 무포지션으로 시작.
            self.positions.clear()
            self.state.clear_positions()
            self.state.set_balance(self.sim_market.balance)
            self._emit_log("INFO", "system", "SIM 모드: 합성 시장 + 페이퍼 잔고 10,000 USDT (무포지션 시작)")
            return
        # LIVE 모드: 실제 거래소/알림 구성.
        from exchange import create_exchange, diagnose_exchange, load_markets_safe
        from notifier import TelegramNotifier

        self.exchange = create_exchange()
        try:
            diag_lines = await diagnose_exchange(self.exchange)
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="diagnose_exchange")
            diag_lines = [f"진단 중단: {format_exception_brief(exc)}"]
        for line in diag_lines:
            log.info("연결 진단 | %s", line)
            self._emit_log("INFO", "system", f"연결 진단: {line}")

        markets_ok, market_err = await load_markets_safe(self.exchange)
        if not markets_ok:
            self._emit_log("ERROR", "system", f"마켓 로드 실패:\n{market_err or 'unknown'}")
        self.notifier = TelegramNotifier()
        self.engine = TradingEngine(self.exchange, notifier=self.notifier)

        # ---- 시작 시 잔여 포지션 정리(클린 스타트) ----
        # entry 만 있고 exit 가 없는(봇이 추적하지 않는) 거래소 잔여 포지션은
        # 외부에서 수동 청산된 것으로 간주하고 시장가로 정리해 무포지션으로 시작한다.
        try:
            closed = await self.engine.flatten_all()
            if closed:
                self._emit_log(
                    "WARNING", "system",
                    f"시작 정리: 잔여 오픈 포지션 청산({len(closed)}개) → {', '.join(closed)}",
                )
            else:
                self._emit_log("INFO", "system", "시작 정리: 잔여 오픈 포지션 없음 (무포지션 시작)")
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="startup_flatten")
            self._emit_log("WARNING", "system", f"시작 정리 중 경고: {exc}")
        self.positions.clear()
        self.state.clear_positions()

        bal, bal_err = await self.engine.fetch_balance_usdt()
        self.state.set_balance(bal)
        if bal_err:
            self._emit_log("ERROR", "system", f"잔고 조회 실패:\n{bal_err}")
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

    # ---- 월간 자동 파인튜닝(재학습) 루프 ----
    async def _finetune_loop(self) -> None:
        """주기(기본 30일)마다 또는 수동 트리거 시 FinBERT를 재학습한다.

        학습은 별도 스레드에서 수행하며(블로킹 방지), 성공 시 감성 모델을
        새 가중치로 핫스왑한다. 실패는 흡수되어 트레이딩 루프를 막지 않는다.
        """
        if not settings.finetune_enabled:
            self._emit_log("INFO", "system", "자동 재학습 비활성화됨 (FINETUNE_ENABLED=false)")
            return
        # 시작 직후 폭주 방지를 위해 잠시 대기.
        await asyncio.sleep(10)
        while self._running:
            try:
                manual = self._finetune_now
                if manual:
                    self._finetune_now = False
                if manual or finetune.due_for_run():
                    trigger = "수동" if manual else "월간"
                    self._emit_log(
                        "INFO", "system",
                        f"FinBERT 재학습 시작({trigger}) — 누적 샘플 {finetune.sample_count()}건",
                    )
                    ok = await asyncio.to_thread(finetune.run_finetune)
                    if ok:
                        await asyncio.to_thread(self.news.sentiment.reload)
                        self._emit_log("INFO", "system", "FinBERT 재학습 완료 — 새 모델 적용됨")
                    else:
                        self._emit_log(
                            "WARNING", "system",
                            "FinBERT 재학습 건너뜀(샘플 부족 또는 오류) — 로그 확인",
                        )
            except Exception as exc:  # noqa: BLE001 - 루프 생존
                log_exception(log, exc, context="finetune_loop")
            # 30초 간격으로 트리거/주기 확인(정지 시 빠르게 빠져나옴).
            await asyncio.sleep(30)

    def trigger_finetune(self) -> None:
        """GUI 등 외부에서 즉시 재학습을 요청한다(다음 루프 틱에 실행)."""
        self._finetune_now = True

    async def _news_context(self, news: str, score: float | None = None) -> str:
        """로그용 뉴스 요약(영문 원문 + 한글 번역)."""
        ko = await asyncio.to_thread(translate_to_korean, news)
        en = news if len(news) <= 120 else news[:117] + "…"
        ko_show = ko if len(ko) <= 120 else ko[:117] + "…"
        if score is not None:
            return f"뉴스({score:+.2f}) | EN: {en} | 한글: {ko_show}"
        return f"EN: {en} | 한글: {ko_show}"

    def _news_entry_allowed(self) -> bool:
        """시작 후 grace 기간이 지났는지(진입 허용 여부)."""
        if self._started_at is None:
            return False
        elapsed = (_now() - self._started_at).total_seconds()
        return elapsed >= settings.news_entry_grace_seconds

    def _is_fresh_entry_news(self, item: AnalyzedNews) -> bool:
        """봇 시작·grace 이후 발행된, 너무 오래되지 않은 기사만 진입 허용."""
        if self._started_at is None:
            return False
        pub = item.item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        cutoff = self._started_at
        if pub < cutoff:
            return False
        max_age = settings.news_max_age_minutes
        if max_age > 0 and (_now() - pub).total_seconds() > max_age * 60:
            return False
        return True

    # ---- 뉴스 콜백(진입 트리거) ----
    async def _on_news(self, item: AnalyzedNews) -> None:
        triggered_at = _now()
        pub = item.item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        title_ko = await asyncio.to_thread(translate_to_korean, item.title)
        self.state.add_news(
            NewsView(
                time=format_kst(pub, "%H:%M:%S"),
                title=item.title,
                score=item.score,
                label=item.label,
                source=item.item.source,
                title_ko=title_ko,
                at_ms=int(triggered_at.timestamp() * 1000),
                published_at_ms=int(pub.timestamp() * 1000),
            )
        )
        # 월간 재학습용 샘플 누적(현재 모델이 부여한 감성 라벨을 정답으로 기록).
        try:
            finetune.record_sample(item.title, item.label)
        except Exception as exc:  # noqa: BLE001
            log.debug("record_sample skipped | %s: %s", type(exc).__name__, exc)

        symbols = detect_symbols(item.title, self.symbols)
        for sym in symbols:
            self._latest_news[sym] = (item.title, item.score)

        # 강한 감성 뉴스만 진입 평가(임계값은 실시간 설정을 사용).
        if abs(item.score) < settings.news_score_threshold:
            return
        if not self._news_entry_allowed():
            return
        if not self._is_fresh_entry_news(item):
            pub_kst = format_kst(pub, "%H:%M:%S")
            self._emit_log(
                "INFO", "entry",
                f"진입 스킵(구기사): 발행 {pub_kst} · {item.title[:80]}",
            )
            return
        for sym in symbols:
            if sym in self.positions:
                await self._maybe_add(sym, item.title, item.score)
            else:
                await self._maybe_enter(sym, item.title, item.score, triggered_at)

    async def _maybe_enter(
        self, symbol: str, news: str, score: float, news_triggered_at: datetime | None = None,
    ) -> None:
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

        # ---- 방향/레버리지 결정 ----
        if settings.auto_leverage:
            # 자동 레버리지: 뉴스 점수 강도로 레버리지를 정하고,
            # |점수|=1.0 이면 역방향이라도 2배로 진입(추세 필터 무시).
            side, leverage = auto_leverage_decision(score, ind.slope)
            if side is None:
                ctx = await self._news_context(news, score)
                self._emit_log(
                    "INFO", "entry",
                    f"진입 스킵 {symbol}: {ctx} · 자동레버리지 기준 미충족 "
                    f"(기울기 {ind.slope:+.4f})",
                )
                return
        else:
            # 수동 레버리지: 뉴스 방향과 기울기 방향이 일치할 때만 진입.
            side = None
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
            leverage = settings.leverage

        await self._open(symbol, side, ind, news, score, leverage, news_triggered_at)

    # ---- 진입 ----
    async def _open(
        self,
        symbol: str,
        side: Side,
        ind,
        news: str,
        score: float,
        leverage: int,
        news_triggered_at: datetime | None = None,
    ) -> None:
        # 실시간 설정 반영: 진입 시점의 명목금액을 사용(진입 후 스냅샷 고정).
        notional = settings.position_size_usdt
        leverage = max(1, int(leverage))
        triggered = news_triggered_at or _now()
        price = ind.last_price
        if self.sim:
            amount = notional / price
            margin = notional / leverage
            self.sim_market.balance -= margin
            self.state.set_balance(self.sim_market.balance)
            order_price = price
            filled = amount
        else:
            result = await self.engine.enter_position(
                symbol, side, leverage=leverage, notional=notional
            )
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
                news_triggered_at=triggered,
                notional=notional, leverage=leverage,
                margin=notional / leverage,
            )
            pos.prev_rsi = ind.rsi
            self.positions[symbol] = pos

        ctx = await self._news_context(news, score)
        self._emit_log(
            "INFO", "entry",
            f"진입 {side.upper()} {symbol} | 진입가={order_price:.4f} 수량={filled:.6f} "
            f"금액={notional:.2f}USDT 레버리지={leverage}x | {ctx}",
        )
        self._sync_position_view(self.positions[symbol])

        # 텔레그램 알림(LIVE 모드, 실제 자격증명 시).
        if self.notifier is not None:
            news_ko = await asyncio.to_thread(translate_to_korean, news)
            await self.notifier.send_position_open(
                symbol=symbol, side=side, amount_usdt=notional,
                entry_price=order_price, news=news, score=score,
                news_ko=news_ko, leverage=leverage,
            )

    # ---- 추가 진입(피라미딩) ----
    async def _maybe_add(self, symbol: str, news: str, score: float) -> None:
        """보유 중인 포지션에 같은 방향·더 강한 뉴스가 오면 1회 추가 진입한다.

        조건: ① 아직 추가 진입한 적 없음 ② 뉴스 방향이 보유 방향과 동일
        ③ 새 점수 강도가 진입 점수보다 큼 ④ 가격이 진입 방향으로 유리하게 이동.
        충족 시 동일 명목금액을 추가하고 평균단가/손익절 라인을 재계산한다.
        """
        pos = self.positions.get(symbol)
        if pos is None or pos.added:
            return
        # 방향 일치 여부(같은 방향 강세/약세 뉴스인지).
        if (pos.side == "long" and score <= 0) or (pos.side == "short" and score >= 0):
            return
        # 더 강한 확신(절대 점수가 진입 시보다 큼)인지.
        if abs(score) <= abs(pos.entry_score):
            return

        ind = await self._indicators(symbol)
        if ind is None:
            return
        favorable = (
            (pos.side == "long" and ind.last_price > pos.entry_price)
            or (pos.side == "short" and ind.last_price < pos.entry_price)
        )
        if not favorable:
            ctx = await self._news_context(news, score)
            self._emit_log(
                "INFO", "entry",
                f"추가진입 보류 {symbol}: 가격 미유리(현재 {ind.last_price:.4f} / "
                f"평균 {pos.entry_price:.4f}) | {ctx}",
            )
            return

        # 추가 분 레버리지: 자동 모드는 점수 강도 기준, 수동 모드는 +1배.
        if settings.auto_leverage:
            add_lev = score_to_leverage(score)
        else:
            add_lev = min(pos.leverage + 1, 25)
        await self._add(pos, ind, news, score, max(1, int(add_lev)))

    async def _add(self, pos: Position, ind, news: str, score: float, leverage: int) -> None:
        notional = settings.position_size_usdt
        leverage = max(1, int(leverage))
        price = ind.last_price
        add_margin = notional / leverage
        if self.sim:
            filled = notional / price
            add_price = price
            self.sim_market.balance -= add_margin
            self.state.set_balance(self.sim_market.balance)
        else:
            result = await self.engine.increase_position(
                pos.symbol, pos.side, leverage=leverage, notional=notional
            )
            if not result.is_filled:
                ctx = await self._news_context(news, score)
                self._emit_log(
                    "ERROR", "order",
                    f"추가진입 실패 {pos.symbol} {pos.side}: {result.reason} | {ctx}",
                )
                return
            add_price = result.price
            filled = result.filled_amount

        async with self._lock:
            prev_entry = pos.entry_price
            atr = ind.atr if ind.atr and not np.isnan(ind.atr) else pos.atr
            pos.add_fill(
                add_amount=filled, add_price=add_price, add_notional=notional,
                add_margin=add_margin, leverage=leverage, atr=atr,
            )
            pos.prev_rsi = ind.rsi

        ctx = await self._news_context(news, score)
        self._emit_log(
            "INFO", "entry",
            f"➕ 추가진입 {pos.side.upper()} {pos.symbol} | 추가가={add_price:.4f} "
            f"수량+={filled:.6f} 평균가={prev_entry:.4f}→{pos.entry_price:.4f} "
            f"총금액={pos.notional:.2f}USDT 배율={leverage}x | {ctx}",
        )
        self._sync_position_view(pos)

        if self.notifier is not None:
            news_ko = await asyncio.to_thread(translate_to_korean, news)
            await self.notifier.send_position_open(
                symbol=pos.symbol, side=pos.side, amount_usdt=notional,
                entry_price=add_price, news=news, score=score,
                news_ko=news_ko, leverage=leverage,
            )

    # ---- 모니터 루프(가격 갱신 + 청산 판정) ----
    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._monitor_once()
            except Exception as exc:  # noqa: BLE001 - 루프 생존
                log_exception(log, exc, context="monitor_loop")
            await asyncio.sleep(max(1, int(settings.monitor_interval)))

    async def _monitor_once(self) -> None:
        # ---- 수동 청산 요청 처리(GUI 버튼) ----
        for req_sym in self.state.pop_close_requests():
            pos = self.positions.get(req_sym)
            if pos is not None:
                await self._close(
                    pos,
                    ExitSignal(
                        True, reason="GUI 수동 청산 요청",
                        exit_type="manual", order_type="market",
                    ),
                )

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
        # 손익 금액(USDT) = 진입 명목금액 × 손익률. (레버리지는 증거금에만 영향)
        pnl_usdt = pos.notional * (pnl_pct / 100)
        if self.sim:
            # 누적 증거금(피라미딩 포함) + 손익을 페이퍼 잔고로 환원.
            margin = pos.margin if pos.margin > 0 else pos.notional / pos.leverage
            self.sim_market.balance += margin + pnl_usdt
            self.state.set_balance(self.sim_market.balance)
        else:
            result = await self.engine.close_position(
                pos.symbol, pos.side, pos.amount, order_type=signal.order_type
            )
            if result.is_filled and result.price:
                exit_price = result.price
                pnl_pct = pos.unrealized_pct()
                pnl_usdt = pos.notional * (pnl_pct / 100)

        async with self._lock:
            self.positions.pop(pos.symbol, None)
        self.state.remove_position(pos.symbol)

        # 수익률 통계용 거래 기록.
        self.state.record_trade({
            "symbol": pos.symbol,
            "side": pos.side,
            "leverage": pos.leverage,
            "notional": pos.notional,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usdt": pnl_usdt,
            "exit_type": signal.exit_type,
            "reason": signal.reason,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": _now().isoformat(),
            "added": pos.added,
        })

        ctx = await self._news_context(pos.entry_news, pos.entry_score)
        self._emit_log(
            "INFO", "exit",
            f"청산 {pos.side.upper()} {pos.symbol} | 진입가={pos.entry_price:.4f} "
            f"청산가={exit_price:.4f} 손익={pnl_pct:+.2f}% ({pnl_usdt:+.2f}USDT) "
            f"사유={signal.exit_type} ({signal.reason}) | {ctx}",
        )

        if self.notifier is not None:
            news_ko = await asyncio.to_thread(translate_to_korean, pos.entry_news)
            await self.notifier.send_position_close(
                symbol=pos.symbol, side=pos.side, amount_usdt=pos.notional,
                entry_price=pos.entry_price, exit_price=exit_price, pnl_pct=pnl_pct,
                reason=f"{signal.exit_type}: {signal.reason}",
                news=pos.entry_news, score=pos.entry_score,
                pnl_usdt=pnl_usdt, news_ko=news_ko,
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
                opened_at=format_kst(pos.opened_at),
                leverage=pos.leverage,
                opened_at_ms=int(pos.opened_at.timestamp() * 1000),
                news_triggered_at_ms=int(pos.news_triggered_at.timestamp() * 1000),
                added=pos.added,
                notional=pos.notional,
                trailing_active=pos.trailing_active,
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
