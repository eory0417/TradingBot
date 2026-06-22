"""Streamlit 웹 GUI 대시보드 (4단계).

실행:  python -m streamlit run app.py
"""

from __future__ import annotations

import asyncio
import html
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bot as botmod
import chart_data
import finetune
from kst_util import KST, TZ_LABEL, format_gui_hms, format_kst, format_legacy_stored_hms, format_ms_kst, ms_to_kst_pandas, now_kst, series_ms_to_kst_pandas, to_kst
from config import settings
from state import STATE as BOT_STATE

st.set_page_config(
    page_title="뉴스 트레이딩 봇 Plus",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 0.6rem; padding-bottom: 0.4rem; max-width: 100%; }
    h1 { font-size: 1.25rem !important; line-height: 1.2 !important;
         margin: 0 0 0.35rem 0 !important; padding: 0 !important; }
    h2, h3, [data-testid="stHeader"] { font-size: 0.92rem !important;
         margin: 0.25rem 0 0.15rem 0 !important; padding: 0 !important; }
    [data-testid="stMetric"] {
        background: rgba(38, 43, 56, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 0.2rem 0.45rem !important;
    }
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #8b949e !important; }
    [data-testid="stCaptionContainer"] p, .stCaption {
        font-size: 0.75rem !important; margin-bottom: 0.1rem !important;
        color: #8b949e !important;
    }
    .news-item {
        font-size: 0.86rem; line-height: 1.45; margin-bottom: 0.45rem;
        color: #e6edf3;
    }
    .news-ko { color: #b0b8c4; font-size: 0.84rem; }
    .news-meta { color: #8b949e; font-size: 0.76rem; display: block; margin-bottom: 0.15rem; }
    .news-meta b { color: #a8b3cf; font-weight: 600; }
    .news-item sub { color: #8b949e; }
    .news-item code {
        background: rgba(110, 118, 129, 0.2); color: #c9d1d9;
        padding: 0.05rem 0.25rem; border-radius: 3px;
    }
    .log-item {
        font-size: 0.84rem; line-height: 1.4; margin-bottom: 0.3rem;
        color: #e6edf3; padding: 0.12rem 0.4rem; border-radius: 4px;
        border-left: 3px solid transparent;
    }
    .log-item code {
        background: rgba(110, 118, 129, 0.2); color: #c9d1d9;
        padding: 0.05rem 0.25rem; border-radius: 3px;
    }
    .log-entry { background: rgba(63,185,80,0.12); border-left-color: #3fb950; }
    .log-exit-win { background: rgba(88,166,255,0.14); border-left-color: #58a6ff; }
    .log-exit-loss { background: rgba(248,81,73,0.12); border-left-color: #f85149; }
    .log-fail { background: rgba(248,81,73,0.16); border-left-color: #f85149; }
    .log-badge {
        display: inline-block; font-size: 0.68rem; font-weight: 700;
        padding: 0.02rem 0.35rem; border-radius: 3px; margin-right: 0.25rem;
        color: #0d1117;
    }
    .badge-entry { background: #3fb950; }
    .badge-exit { background: #58a6ff; }
    .badge-fail { background: #f85149; color: #fff; }
    .pos-row { font-size: 0.82rem; }
    .bot-status-hint {
        font-size: 0.68rem; color: #8b949e; margin-top: -0.35rem;
        line-height: 1.2; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="column"] { padding: 0 0.25rem !important; }
    hr { margin: 0.35rem 0 !important; border-color: rgba(255,255,255,0.08) !important; }
</style>
""",
    unsafe_allow_html=True,
)

_NEWS_LOG_HEIGHT = 230
_CHART_HEIGHT = 210
_PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161B22",
    font=dict(color="#E6EDF3", size=10),
)


class BotRunner:
    """봇 스레드 + UI가 공유하는 단일 BotState를 묶는다."""

    def __init__(self, state) -> None:
        self.state = state
        self.thread: threading.Thread | None = None
        self.bot: botmod.TradingBot | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.is_alive():
            return
        self.bot = botmod.TradingBot(state=self.state)
        self._loop = None

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self.bot.run())
            except Exception as exc:  # noqa: BLE001
                self.state.log("ERROR", "system", f"봇 스레드 오류: {exc}")
                self.state.set_running(False, status="error")
            finally:
                self._loop = None
                loop.close()

        self.thread = threading.Thread(target=_run, daemon=True, name="trading-bot")
        self.thread.start()

    def stop(self) -> None:
        if self.bot is not None:
            self.bot.stop()
            self.bot.positions.clear()
        self.state.clear_positions()
        self.state.pop_close_requests()
        self.state.set_running(False, status="stopped")


@st.cache_resource
def get_runner(_state_api_version: int = 3) -> BotRunner:
    """봇 실행기 싱글톤. _state_api_version 을 올리면 캐시가 갱신된다."""
    # 프로세스( streamlit run ) 시작 시 이전 세션 잔여 포지션 표시를 제거.
    BOT_STATE.clear_positions()
    BOT_STATE.pop_close_requests()
    return BotRunner(BOT_STATE)


runner = get_runner()
STATE = runner.state  # == BOT_STATE (모듈 싱글톤과 동일)


def _reset_local_positions() -> None:
    """로컬 오픈 포지션(UI·봇 내부)을 모두 비운다.

    바이낸스에서 수동 청산했거나 이전 로그/세션 잔여가 있어도,
    봇이 동작 중이 아닐 때는 항상 무포지션으로 맞춘다.
    """
    STATE.clear_positions()
    STATE.pop_close_requests()
    if runner.bot is not None:
        runner.bot.positions.clear()


def _sym_key(symbol: str) -> str:
    """Streamlit 위젯 key용 — '/' 등 특수문자 제거."""
    return symbol.replace("/", "_").replace(":", "_").replace(" ", "_")


def _handle_manual_close(symbol: str) -> None:
    """GUI 수동 청산 — 봇 실행 중이면 즉시 청산, 아니면 화면 상태만 정리."""
    bot = runner.bot
    if runner.is_alive() and bot is not None:
        pos = bot.positions.get(symbol)
        if pos is not None:
            if runner._loop is not None:  # noqa: SLF001
                from strategy import ExitSignal

                asyncio.run_coroutine_threadsafe(
                    bot._close(
                        pos,
                        ExitSignal(
                            True,
                            reason="GUI 수동 청산 요청",
                            exit_type="manual",
                            order_type="market",
                        ),
                    ),
                    runner._loop,
                )
                STATE.log("INFO", "system", f"🧹 {symbol} 수동 청산 요청 — 처리 중")
            else:
                STATE.request_close(symbol)
                STATE.log("INFO", "system", f"🧹 {symbol} 수동 청산 요청 — 루프 대기")
            return
        STATE.remove_position(symbol)
        STATE.log("WARNING", "system", f"🧹 {symbol} 표시 포지션 제거(봇 내부 없음)")
        return

    STATE.remove_position(symbol)
    if bot is not None:
        bot.positions.pop(symbol, None)
    STATE.log("INFO", "system", f"🧹 {symbol} 수동 제거 (봇 정지 — 표시만 초기화)")


def _bot_status_compact() -> tuple[str, str]:
    """상단 metric용 (짧은 라벨, 보조 설명)."""
    if runner.is_alive():
        if STATE.status.startswith("running"):
            return "🟢 실행 중", STATE.status
        if STATE.status.startswith("시작"):
            return "🟡 시작 중", "FinBERT·거래소 연결"
        return "🔵 동작 중", STATE.status
    if STATE.running:
        return "🟡 기동 중", "스레드 대기"
    return "⚪ 대기", "▶ 시작 필요"


def _news_time_meta(nw) -> str:
    """뉴스 항목 메타: 발행·수신 시각(KST) + RSS 출처."""
    pub_ms = int(getattr(nw, "published_at_ms", 0) or 0)
    recv_ms = int(getattr(nw, "at_ms", 0) or 0)
    pub_hms = (
        format_gui_hms(at_ms=pub_ms)
        if pub_ms
        else format_gui_hms(stored=getattr(nw, "time", "") or "")
    ) or "—"
    recv_hms = format_gui_hms(at_ms=recv_ms) if recv_ms else "—"
    source = html.escape((getattr(nw, "source", "") or "").strip()) or "—"
    return (
        f'<span class="news-meta">'
        f"발행 <code>{pub_hms}</code> · 수신 <code>{recv_hms}</code> {TZ_LABEL}"
        f' · 출처 <b>{source}</b></span>'
    )


def render_sidebar() -> None:
    st.sidebar.header("⚙️ 설정")

    # 증거금 모드는 항상 isolated 로 고정(선택 UI 제거).
    margin_mode = "isolated"
    manual_leverage = st.sidebar.checkbox(
        "수동 레버리지", value=not settings.auto_leverage,
        help="체크: 슬라이더로 직접 설정 · 해제: 뉴스 점수로 자동 설정",
    )
    auto_leverage = not manual_leverage
    if manual_leverage:
        leverage = st.sidebar.slider("레버리지", 1, 25, settings.leverage)
    else:
        leverage = settings.leverage  # 자동 모드: 진입 시 점수로 결정
        st.sidebar.caption(
            "자동 레버리지: |점수| 0.7→1x · 0.8→2x · 0.9→3x · 1.0→4x "
            "(점수 1.0·역방향이면 2x 진입)"
        )
    notional = st.sidebar.number_input(
        "진입금 (USDT)", min_value=5.0, value=float(settings.position_size_usdt), step=5.0
    )
    stop_loss = st.sidebar.number_input(
        "손절 (%)", min_value=0.1, value=float(settings.stop_loss_pct), step=0.1
    )
    trailing_profit = st.sidebar.number_input(
        "Trailing 이익구간 (%)",
        min_value=0.0,
        value=float(settings.trailing_profit_pct),
        step=0.1,
        help="미실현 이익이 이 값 이상일 때만 Trailing 익절이 활성화됩니다.",
    )
    atr_mult = st.sidebar.number_input(
        "Trailing ATR", min_value=0.5, value=float(settings.trailing_atr_mult), step=0.5
    )
    time_exit = st.sidebar.number_input(
        "시간청산 (h)", min_value=0.5, value=float(settings.time_exit_hours), step=0.5
    )

    settings.margin_mode = margin_mode
    settings.auto_leverage = bool(auto_leverage)
    settings.leverage = int(leverage)
    settings.position_size_usdt = float(notional)
    settings.stop_loss_pct = float(stop_loss)
    settings.trailing_profit_pct = float(trailing_profit)
    settings.trailing_atr_mult = float(atr_mult)
    settings.time_exit_hours = float(time_exit)
    STATE.update_settings(
        margin_mode=margin_mode, auto_leverage=bool(auto_leverage),
        leverage=int(leverage), notional=float(notional),
        stop_loss_pct=float(stop_loss), trailing_profit_pct=float(trailing_profit),
        trailing_atr_mult=float(atr_mult),
        time_exit_hours=float(time_exit),
    )

    st.sidebar.divider()
    col1, col2 = st.sidebar.columns(2)
    running = runner.is_alive() or STATE.running
    if col1.button(
        "▶ 시작", width="stretch", type="primary", disabled=running,
    ):
        mode = botmod.exchange_mode_label()
        STATE.set_running(True, status=f"시작 중 ({mode})")
        STATE.log("INFO", "system", "▶ 시작 버튼 — 봇 초기화 중 (FinBERT·거래소 연결)")
        runner.start()
        st.rerun()
    if col2.button("■ 정지", width="stretch", disabled=not running):
        runner.stop()
        STATE.log("INFO", "system", "■ 정지 버튼 — 봇 종료 요청")
        st.rerun()

    mode = botmod.exchange_mode_label()
    if runner.is_alive():
        thread_label = "🟢 스레드 실행 중"
    elif STATE.running:
        thread_label = "🟡 시작 처리 중"
    else:
        thread_label = "⚪ 대기"
    st.sidebar.caption(f"{thread_label} · 모드: **{mode}** · **{STATE.status}**")
    if botmod.has_real_credentials() and settings.binance_testnet:
        st.sidebar.caption("Demo: demo.binance.com 키 필요. 실거래는 BINANCE_TESTNET=false")
    st.sidebar.caption("ℹ️ 설정 변경은 즉시 반영되며, 진행 중 포지션이 아닌 '앞으로의 진입'에만 적용됩니다.")

    st.sidebar.divider()
    st.sidebar.caption(f"🧠 FinBERT 재학습 · 누적 샘플 {finetune.sample_count()}건")
    bot = getattr(runner, "bot", None)
    if st.sidebar.button(
        "🔄 모델 재학습(수동)", width="stretch",
        disabled=bot is None, help="월 1회 자동 재학습. 지금 즉시 실행하려면 클릭.",
    ):
        if bot is not None:
            bot.trigger_finetune()
            STATE.log("INFO", "system", "🔄 수동 재학습 요청 — 다음 점검 주기에 실행됩니다.")
            st.toast("재학습을 요청했습니다. 로그에서 진행 상황을 확인하세요.")


def _collect_news_markers(symbol: str, pos) -> list[dict]:
    """팝업 차트용 뉴스 인식 시점 마커 목록."""
    markers: list[dict] = []
    seen_ms: set[int] = set()

    if pos is not None and getattr(pos, "news_triggered_at_ms", 0):
        markers.append(
            {
                "at_ms": int(pos.news_triggered_at_ms),
                "label": "뉴스인식",
                "score": float(pos.entry_score),
                "title": (pos.entry_news or "진입 트리거 뉴스")[:80],
            }
        )
        seen_ms.add(int(pos.news_triggered_at_ms))
    elif pos is not None and getattr(pos, "entry_news", ""):
        # 구 포지션 등 news_triggered_at_ms 미기록 시 진입 뉴스를 진입 시각에 표시.
        at_ms = int(getattr(pos, "opened_at_ms", 0) or 0)
        if at_ms:
            markers.append(
                {
                    "at_ms": at_ms,
                    "label": "뉴스인식",
                    "score": float(pos.entry_score),
                    "title": pos.entry_news[:80],
                }
            )
            seen_ms.add(at_ms)

    for nw in STATE.get_news(50):
        at_ms = int(getattr(nw, "at_ms", 0) or 0)
        if not at_ms or at_ms in seen_ms:
            continue
        if symbol in botmod.detect_symbols(nw.title, [symbol]):
            markers.append(
                {
                    "at_ms": at_ms,
                    "label": "뉴스",
                    "score": float(nw.score),
                    "title": (nw.title_ko or nw.title)[:80],
                }
            )
            seen_ms.add(at_ms)

    return sorted(markers, key=lambda m: m["at_ms"])


def _same_moment_ms(a_ms: int, b_ms: int, window_ms: int = 60_000) -> bool:
    """두 시각이 같은 캔들/분봉 안에 있는지(겹침 판정)."""
    return abs(a_ms - b_ms) <= window_ms


def _price_band(ohlcv_df: pd.DataFrame, pos) -> tuple[float, float]:
    """캔들 + SL/진입/익절을 모두 담는 Y축 범위."""
    y_lo = float(ohlcv_df["low"].min())
    y_hi = float(ohlcv_df["high"].max())
    levels = [y_lo, y_hi]
    if pos is not None:
        levels.extend([pos.stop_loss, pos.entry_price])
        if getattr(pos, "trailing_active", False):
            levels.append(pos.trailing_stop)
    y_min, y_max = min(levels), max(levels)
    pad = max((y_max - y_min) * 0.06, y_max * 0.0005)
    return y_min - pad, y_max + pad


def _add_price_line(
    fig: go.Figure,
    *,
    y: float,
    label: str,
    color: str,
    dash: str,
    x_end,
    y_min: float,
    y_max: float,
) -> None:
    """가격 수평선 + 차트 안쪽에 보이는 라벨."""
    fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=1.2)
    # SL/Entry 가 캔들 밖이어도 라벨이 잘리지 않도록 Y 위치를 클램프.
    label_y = max(y_min, min(y, y_max))
    fig.add_annotation(
        x=x_end,
        y=label_y,
        text=label,
        showarrow=False,
        xanchor="right",
        xshift=-4,
        yanchor="middle",
        font=dict(color=color, size=10),
        bgcolor="rgba(22,27,34,0.88)",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
    )


def build_candlestick_figure(
    symbol: str,
    ohlcv: list,
    *,
    height: int = _CHART_HEIGHT,
    pos=None,
    news_markers: list[dict] | None = None,
    timeframe: str = "15m",
    show_legend: bool = False,
) -> go.Figure | None:
    """OHLCV 데이터로 캔들 차트를 그린다. 진입·뉴스 시점 마커를 선택적으로 표시."""
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = series_ms_to_kst_pandas(df["ts"])
    y_min, y_max = _price_band(df, pos)
    y_lo = float(df["low"].min())
    y_hi = float(df["high"].max())
    x_end = df["time"].iloc[-1]

    fig = go.Figure(
        data=[go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=symbol,
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        )]
    )

    if pos is not None:
        _add_price_line(
            fig, y=pos.stop_loss, label=f"SL {pos.stop_loss:.4f}",
            color="#f85149", dash="dash", x_end=x_end, y_min=y_min, y_max=y_max,
        )
        if getattr(pos, "trailing_active", False):
            _add_price_line(
                fig, y=pos.trailing_stop, label=f"TP {pos.trailing_stop:.4f}",
                color="#3fb950", dash="dot", x_end=x_end, y_min=y_min, y_max=y_max,
            )
        _add_price_line(
            fig, y=pos.entry_price, label=f"Entry {pos.entry_price:.4f}",
            color="#58a6ff", dash="solid", x_end=x_end, y_min=y_min, y_max=y_max,
        )

        entry_ms = int(getattr(pos, "opened_at_ms", 0) or 0)
        if entry_ms:
            entry_dt = ms_to_kst_pandas(entry_ms)
            span = y_max - y_min
            entry_marker_y = y_min + span * 0.05  # 차트 하단 여백 — 캔들과 겹치지 않음
            fig.add_trace(go.Scatter(
                x=[entry_dt, entry_dt], y=[y_min, y_max], mode="lines",
                line=dict(color="#d29922", width=1.2, dash="dot"),
                name="진입시각", hoverinfo="skip", showlegend=show_legend,
            ))
            # 진입가까지 얇은 연결선(가격 위치는 파란 Entry 라인으로 확인).
            fig.add_trace(go.Scatter(
                x=[entry_dt, entry_dt], y=[entry_marker_y, pos.entry_price],
                mode="lines",
                line=dict(color="#d29922", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[entry_dt], y=[entry_marker_y], mode="markers+text",
                marker=dict(symbol="star", size=13, color="#d29922",
                            line=dict(color="#ffffff", width=1)),
                text=["진입"], textposition="bottom center",
                textfont=dict(color="#d29922", size=10),
                name="진입", hoverinfo="text",
                hovertext=[f"진입 {pos.opened_at} ({TZ_LABEL}) @ {pos.entry_price:.4f}"],
                showlegend=show_legend,
            ))

    entry_ms = int(getattr(pos, "opened_at_ms", 0) or 0) if pos is not None else 0
    span = y_max - y_min
    news_y_high = y_min + span * 0.92

    for marker in news_markers or []:
        at_ms = int(marker.get("at_ms", 0))
        if not at_ms:
            continue
        news_dt = ms_to_kst_pandas(at_ms)
        score = marker.get("score", 0.0)
        title = marker.get("title", "")
        label = marker.get("label", "뉴스")
        overlap_entry = entry_ms and _same_moment_ms(at_ms, entry_ms)
        marker_y = news_y_high if overlap_entry else (y_hi * 0.85 + y_lo * 0.15)

        fig.add_trace(go.Scatter(
            x=[news_dt, news_dt], y=[y_min, y_max], mode="lines",
            line=dict(color="#a371f7", width=1.4, dash="dash"),
            name=f"{label}시각", hoverinfo="skip", showlegend=show_legend,
        ))
        fig.add_trace(go.Scatter(
            x=[news_dt], y=[marker_y], mode="markers+text",
            marker=dict(symbol="diamond", size=13, color="#a371f7",
                        line=dict(color="#ffffff", width=1)),
            text=[label], textposition="top center",
            textfont=dict(color="#a371f7", size=10),
            name=label, hoverinfo="text",
            hovertext=[
                f"{label} {news_dt.strftime('%H:%M:%S')} ({TZ_LABEL}) · score {score:+.2f}\n{title}"
                + ("\n(진입과 동일 시각 — 상단 표시)" if overlap_entry else "")
            ],
            showlegend=show_legend,
        ))

    title_text = symbol if timeframe == "15m" else f"{symbol} · {timeframe}"
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=72, t=28, b=4),
        xaxis_rangeslider_visible=False,
        title=dict(text=title_text, font=dict(size=11, color="#E6EDF3")),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(range=[y_min, y_max], fixedrange=False),
        xaxis=dict(title=f"시간 ({TZ_LABEL})"),
        **_PLOTLY_DARK,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
    return fig


def candlestick(symbol: str, height: int = _CHART_HEIGHT) -> go.Figure | None:
    ohlcv = STATE.get_ohlcv(symbol)
    pos = next((p for p in STATE.get_positions() if p.symbol == symbol), None)
    news_markers = _collect_news_markers(symbol, pos) if pos else []
    return build_candlestick_figure(
        symbol, ohlcv, height=height, pos=pos, news_markers=news_markers,
    )


@st.dialog("📈 차트 상세", width="large")
def chart_dialog() -> None:
    """선택한 코인의 캔들차트를 큰 화면으로 표시한다."""
    sym = st.session_state.get("zoom_symbol")
    if not sym:
        st.caption("표시할 심볼이 없습니다.")
        return

    pos = next((p for p in STATE.get_positions() if p.symbol == sym), None)
    tf_options = list(chart_data.CHART_TIMEFRAMES)
    tf_key = f"chart_tf_{_sym_key(sym)}"
    default_tf = st.session_state.get(tf_key, "15m")
    if default_tf not in tf_options:
        default_tf = "15m"

    tf = st.selectbox(
        "주기",
        tf_options,
        index=tf_options.index(default_tf),
        key=f"chart_tf_select_{_sym_key(sym)}",
        help="1분봉 등으로 바꿔 진입·뉴스 인식 시점이 늦지 않았는지 확인하세요.",
    )
    st.session_state[tf_key] = tf

    with st.spinner(f"{sym} · {tf} 차트 불러오는 중…"):
        ohlcv = chart_data.fetch_ohlcv(sym, tf)
    news_markers = _collect_news_markers(sym, pos)

    trail_note = ""
    if pos is not None:
        if getattr(pos, "trailing_active", False):
            trail_note = f" · Trailing 활성 (익절 {pos.trailing_stop:.4f})"
        else:
            trail_note = (
                f" · Trailing 대기 (이익 {settings.trailing_profit_pct:.1f}%↑ 후 활성)"
            )

    st.markdown(f"**{sym}** · {tf}{trail_note}")
    st.caption(f"◆ 보라(상단) = 뉴스 인식 · ★ 금색(하단) = 진입 · 시간은 {TZ_LABEL}")

    fig = build_candlestick_figure(
        sym, ohlcv, height=620, pos=pos, news_markers=news_markers,
        timeframe=tf, show_legend=True,
    )
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key=f"zoom_chart_{_sym_key(sym)}_{tf}")
    else:
        st.caption("차트 데이터를 불러오지 못했습니다. 잠시 후 다시 시도하세요.")


def _compute_stats(trades: list[dict], start_dt: datetime, end_dt: datetime):
    """기간 내 청산 거래를 코인별/전체로 집계한다."""
    rows = []
    for t in trades:
        try:
            closed = datetime.fromisoformat(t.get("closed_at", ""))
        except ValueError:
            continue
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        closed = to_kst(closed)
        if start_dt <= closed < end_dt:
            rows.append(t)
    return rows


def _position_margin(p) -> float:
    """포지션에 묶인 증거금(USDT)."""
    lev = max(int(getattr(p, "leverage", 1) or 1), 1)
    notional = float(getattr(p, "notional", 0) or 0)
    if notional <= 0:
        return 0.0
    return notional / lev


def _account_summary(free_usdt: float, positions: list) -> dict[str, float]:
    """가용·투입·미실현·총 평가 잔고를 계산한다."""
    used_margin = sum(_position_margin(p) for p in positions)
    unrealized = sum(
        float(getattr(p, "notional", 0) or 0) * float(getattr(p, "unrealized_pct", 0) or 0) / 100
        for p in positions
    )
    equity = free_usdt + used_margin + unrealized
    return {
        "free": free_usdt,
        "used_margin": used_margin,
        "unrealized": unrealized,
        "equity": equity,
    }


@st.dialog("📊 수익률 통계", width="large")
def stats_dialog() -> None:
    """프로그램 시작 시점(기본)부터 현재까지 코인별·전체 수익률을 보여준다."""
    trades = STATE.get_trades()
    open_pos = STATE.get_positions()
    free_balance = STATE.get_balance()
    acct = _account_summary(free_balance, open_pos)

    st.markdown("##### 💰 계좌 잔고")
    if open_pos:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("가용 잔고", f"{acct['free']:,.2f} USDT", help="새 진입에 쓸 수 있는 USDT")
        b2.metric(
            "투입 증거금",
            f"{acct['used_margin']:,.2f} USDT",
            help="오픈 포지션에 묶인 증거금(진입 명목÷레버리지)",
        )
        b3.metric(
            "미실현 손익",
            f"{acct['unrealized']:+,.2f} USDT",
            help="현재가 기준 오픈 포지션 손익",
        )
        b4.metric(
            "총 평가 잔고",
            f"{acct['equity']:,.2f} USDT",
            help="가용 + 투입 증거금 + 미실현 손익",
        )
        st.caption(
            f"총 평가 = 가용 {acct['free']:,.2f} + 증거금 {acct['used_margin']:,.2f} "
            f"+ 미실현 {acct['unrealized']:+,.2f} USDT · 보유 {len(open_pos)}건"
        )
    else:
        st.metric("잔고 (USDT)", f"{acct['free']:,.2f}")
        st.caption("오픈 포지션 없음 — 가용 잔고와 동일")

    st.divider()
    session_start = to_kst(STATE.session_start)
    today = now_kst().date()

    c1, c2 = st.columns(2)
    start_d = c1.date_input("시작일", value=session_start.date())
    end_d = c2.date_input("종료일", value=today)
    start_dt = datetime(start_d.year, start_d.month, start_d.day, tzinfo=KST)
    end_dt = datetime(end_d.year, end_d.month, end_d.day, tzinfo=KST) + timedelta(days=1)
    st.caption(
        f"세션 시작: {session_start.strftime('%Y-%m-%d %H:%M')} {TZ_LABEL} · "
        f"집계 기간: {start_d} ~ {end_d} ({TZ_LABEL})"
    )

    rows = _compute_stats(trades, start_dt, end_dt)

    if rows:
        df = pd.DataFrame(rows)
        total = len(df)
        wins = int((df["pnl_usdt"] > 0).sum())
        total_pnl = float(df["pnl_usdt"].sum())
        winrate = wins / total * 100 if total else 0.0
        avg_pct = float(df["pnl_pct"].mean())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("청산 거래", f"{total} 건")
        m2.metric("승률", f"{winrate:.1f}%", f"{wins}승 {total - wins}패")
        m3.metric("총 손익", f"{total_pnl:+,.2f} USDT")
        m4.metric("평균 손익률", f"{avg_pct:+.2f}%")

        st.markdown("##### 코인별 성과")
        grp = (
            df.groupby("symbol")
            .agg(
                거래수=("pnl_usdt", "count"),
                승=("pnl_usdt", lambda s: int((s > 0).sum())),
                총손익USDT=("pnl_usdt", "sum"),
                평균손익률=("pnl_pct", "mean"),
            )
            .reset_index()
            .rename(columns={"symbol": "코인"})
        )
        grp["승률%"] = (grp["승"] / grp["거래수"] * 100).round(1)
        grp["총손익USDT"] = grp["총손익USDT"].round(2)
        grp["평균손익률"] = grp["평균손익률"].round(2)
        st.dataframe(
            grp[["코인", "거래수", "승", "승률%", "총손익USDT", "평균손익률"]],
            width="stretch", hide_index=True,
        )

        st.markdown("##### 청산 거래 내역")
        hist = df.copy()
        hist["진입"] = hist["entry_price"].round(4)
        hist["청산"] = hist["exit_price"].round(4)
        hist["손익%"] = hist["pnl_pct"].round(2)
        hist["손익USDT"] = hist["pnl_usdt"].round(2)
        hist["시각"] = hist["closed_at"].apply(
            lambda s: (
                format_kst(datetime.fromisoformat(s.replace("Z", "+00:00")))
                if s else ""
            )
        )
        st.dataframe(
            hist[["시각", "symbol", "side", "leverage", "진입", "청산", "손익%", "손익USDT", "exit_type"]]
            .rename(columns={"symbol": "코인", "side": "방향", "leverage": "배율", "exit_type": "사유"}),
            width="stretch", hide_index=True, height=220,
        )
    else:
        st.info("선택한 기간에 청산된 거래가 없습니다.")

    # 현재 보유 중(미실현) 포지션도 참고로 표시.
    if open_pos:
        st.markdown("##### 현재 보유(미실현)")
        odf = pd.DataFrame([{
            "코인": p.symbol, "방향": p.side.upper(), "배율": f"{p.leverage}x",
            "진입": round(p.entry_price, 4), "현재": round(p.mark_price, 4),
            "명목USDT": round(p.notional, 2),
            "증거금USDT": round(_position_margin(p), 2),
            "미실현%": p.unrealized_pct,
            "미실현USDT": round(p.notional * p.unrealized_pct / 100, 2),
        } for p in open_pos])
        st.dataframe(odf, width="stretch", hide_index=True)

    if st.button("닫기", width="stretch"):
        st.rerun()


def _log_style(lg: dict) -> tuple[str, str]:
    """로그 항목의 강조 CSS 클래스와 배지를 결정한다."""
    cat = lg.get("category", "")
    msg = lg.get("message", "")
    level = lg.get("level", "INFO")
    if cat == "entry" and ("진입 " in msg or "추가진입" in msg) and "스킵" not in msg and "보류" not in msg:
        return "log-entry", '<span class="log-badge badge-entry">진입</span>'
    if cat == "exit":
        loss = "손익=-" in msg
        cls = "log-exit-loss" if loss else "log-exit-win"
        return cls, '<span class="log-badge badge-exit">청산</span>'
    if level == "ERROR" or "실패" in msg:
        return "log-fail", '<span class="log-badge badge-fail">실패</span>'
    return "", ""


@st.fragment(run_every=2)
def render_dashboard() -> None:
    # 봇 미실행 시 이전 로그/세션과 무관하게 오픈 포지션 표시를 항상 비운다.
    if not runner.is_alive():
        _reset_local_positions()

    positions = STATE.get_positions()
    bot_label, bot_hint = _bot_status_compact()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("잔고 (USDT)", f"{STATE.get_balance():,.2f}")
    c2.metric("포지션", f"{len(positions)} / {settings.max_positions}")
    c3.metric("상태", STATE.status)
    with c4:
        st.metric("봇", bot_label)
        st.markdown(f'<p class="bot-status-hint">{bot_hint}</p>', unsafe_allow_html=True)

    st.markdown("##### 📌 오픈 포지션")
    if positions:
        _col_w = [1.4, 0.9, 0.7, 1.2, 1.3, 1.3, 1.3, 1.3, 1.0, 1.0]
        _heads = ["코인", "방향", "배율", "수량", "진입", "현재", "손절", "익절", "PnL%", "청산"]
        hcols = st.columns(_col_w)
        for hc, ht in zip(hcols, _heads):
            hc.markdown(f'<div class="pos-row"><b>{ht}</b></div>', unsafe_allow_html=True)
        for p in positions:
            lev = getattr(p, "leverage", 1)
            added_mark = " ➕" if getattr(p, "added", False) else ""
            pnl_color = "#3fb950" if p.unrealized_pct >= 0 else "#f85149"
            side_color = "#3fb950" if p.side == "long" else "#f85149"
            rc = st.columns(_col_w)
            rc[0].markdown(f'<div class="pos-row">{p.symbol}{added_mark}</div>', unsafe_allow_html=True)
            rc[1].markdown(
                f'<div class="pos-row" style="color:{side_color}"><b>{p.side.upper()}</b></div>',
                unsafe_allow_html=True,
            )
            rc[2].markdown(f'<div class="pos-row">{lev}x</div>', unsafe_allow_html=True)
            rc[3].markdown(f'<div class="pos-row">{p.amount:.4f}</div>', unsafe_allow_html=True)
            rc[4].markdown(f'<div class="pos-row">{p.entry_price:.4f}</div>', unsafe_allow_html=True)
            rc[5].markdown(f'<div class="pos-row">{p.mark_price:.4f}</div>', unsafe_allow_html=True)
            rc[6].markdown(f'<div class="pos-row">{p.stop_loss:.4f}</div>', unsafe_allow_html=True)
            if getattr(p, "trailing_active", False):
                rc[7].markdown(f'<div class="pos-row">{p.trailing_stop:.4f}</div>', unsafe_allow_html=True)
            else:
                rc[7].markdown(
                    f'<div class="pos-row" style="color:#8b949e">대기</div>',
                    unsafe_allow_html=True,
                )
            rc[8].markdown(
                f'<div class="pos-row" style="color:{pnl_color}"><b>{p.unrealized_pct:+.2f}%</b></div>',
                unsafe_allow_html=True,
            )
            if rc[9].button("청산", key=f"close_{_sym_key(p.symbol)}", width="stretch"):
                _handle_manual_close(p.symbol)
    else:
        st.caption("오픈 포지션 없음")

    ch_head, ch_btn = st.columns([4, 1])
    ch_head.markdown("##### 📈 차트 (15m)")
    if ch_btn.button("📊 수익률 통계", width="stretch", key="stats_btn"):
        st.session_state["pending_dialog"] = ("stats", None)
        st.rerun()
    open_syms = [p.symbol for p in positions]
    data_syms = STATE.symbols_with_data()
    chart_syms = (open_syms + [s for s in data_syms if s not in open_syms])[:2]
    if chart_syms:
        cols = st.columns(len(chart_syms))
        for col, sym in zip(cols, chart_syms):
            fig = candlestick(sym)
            if fig is not None:
                col.plotly_chart(fig, width="stretch", key=f"chart_{_sym_key(sym)}")
                if col.button("🔍 크게 보기", key=f"zoom_btn_{_sym_key(sym)}", width="stretch"):
                    st.session_state["pending_dialog"] = ("chart", sym)
                    st.rerun()
    else:
        st.caption("차트 데이터 수집 중…")

    st.markdown("##### 📰 뉴스 · 로그")
    n_col, l_col = st.columns(2)

    with n_col:
        st.caption(f"실시간 뉴스 (EN + 한글) · 발행/수신 {TZ_LABEL}")
        news = STATE.get_news(30)
        with st.container(height=_NEWS_LOG_HEIGHT):
            if not news:
                st.caption(
                    "뉴스 수집 중… (FinBERT 로딩 후 워밍업·신규 RSS 기사 표시)"
                )
            for nw in news:
                icon = "🟢" if nw.score > 0.2 else "🔴" if nw.score < -0.2 else "⚪"
                ko = nw.title_ko or nw.title
                st.markdown(
                    f'<div class="news-item">{icon} {_news_time_meta(nw)}<br>'
                    f"<b>[{nw.score:+.2f} {nw.label}]</b> "
                    f"🇺🇸 {html.escape(nw.title)}<br>"
                    f'<span class="news-ko">🇰🇷 {html.escape(ko)}</span></div>',
                    unsafe_allow_html=True,
                )

    with l_col:
        st.caption(f"진입/청산 · 주문 실패 · 시간 {TZ_LABEL}")
        logs = STATE.get_logs(80)
        with st.container(height=_NEWS_LOG_HEIGHT):
            if not logs:
                st.caption("로그 대기 중…")
            for lg in logs:
                cls, badge = _log_style(lg)
                icon = {"ERROR": "🔴", "WARNING": "🟡"}.get(lg["level"], "🟢")
                time_hms = format_gui_hms(
                    at_ms=int(lg.get("time_ms", 0) or 0),
                    stored=str(lg.get("time", "")),
                )
                msg = html.escape(str(lg.get("message", ""))).replace("\n", "<br>")
                st.markdown(
                    f'<div class="log-item {cls}">{badge}{icon} '
                    f'<code>{time_hms}</code> '
                    f"<b>[{lg['category']}]</b> {msg}</div>",
                    unsafe_allow_html=True,
                )


st.markdown("# 📊 바이낸스 USDⓈ-M 뉴스 트레이딩 봇 **Plus**")

render_sidebar()

# 다이얼로그는 메인 스크립트 범위에서 열어야 대시보드 자동 새로고침(fragment)에
# 의해 닫히지 않는다. 버튼이 pending 플래그를 세팅한 뒤 전체 rerun → 여기서 1회 오픈.
_pending = st.session_state.pop("pending_dialog", None)
if _pending:
    _kind, _arg = _pending
    if _kind == "chart":
        st.session_state["zoom_symbol"] = _arg
        chart_dialog()
    elif _kind == "stats":
        stats_dialog()

render_dashboard()
