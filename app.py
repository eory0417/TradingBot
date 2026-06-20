"""Streamlit 웹 GUI 대시보드 (4단계).

실행:  python -m streamlit run app.py
"""

from __future__ import annotations

import asyncio
import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bot as botmod
from config import settings

st.set_page_config(
    page_title="뉴스 트레이딩 봇",
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
        background: rgba(240,242,246,0.45); border-radius: 6px;
        padding: 0.2rem 0.45rem !important;
    }
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
    [data-testid="stAlert"], .stAlert {
        padding: 0.35rem 0.55rem !important; margin: 0.15rem 0 !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stCaptionContainer"] p, .stCaption {
        font-size: 0.75rem !important; margin-bottom: 0.1rem !important;
    }
    .news-item { font-size: 0.78rem; line-height: 1.35; margin-bottom: 0.35rem; }
    .news-ko { color: #444; font-size: 0.76rem; }
    .log-item { font-size: 0.76rem; line-height: 1.3; margin-bottom: 0.25rem; }
    div[data-testid="column"] { padding: 0 0.25rem !important; }
    hr { margin: 0.35rem 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


class BotRunner:
    """봇 스레드 + UI가 공유하는 단일 BotState를 묶는다."""

    def __init__(self, state) -> None:
        self.state = state
        self.thread: threading.Thread | None = None
        self.bot: botmod.TradingBot | None = None

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.is_alive():
            return
        self.bot = botmod.TradingBot(state=self.state)

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.bot.run())
            except Exception as exc:  # noqa: BLE001
                self.state.log("ERROR", "system", f"봇 스레드 오류: {exc}")
                self.state.set_running(False, status="error")
            finally:
                loop.close()

        self.thread = threading.Thread(target=_run, daemon=True, name="trading-bot")
        self.thread.start()

    def stop(self) -> None:
        if self.bot is not None:
            self.bot.stop()
        self.state.set_running(False, status="stopped")


@st.cache_resource
def get_runner() -> BotRunner:
    from state import BotState
    return BotRunner(BotState())


runner = get_runner()
STATE = runner.state


def render_sidebar() -> None:
    st.sidebar.header("⚙️ 설정")

    margin_mode = st.sidebar.selectbox(
        "증거금 모드", ["isolated", "cross"],
        index=0 if settings.margin_mode == "isolated" else 1,
    )
    leverage = st.sidebar.slider("레버리지", 1, 25, settings.leverage)
    notional = st.sidebar.number_input(
        "진입금 (USDT)", min_value=5.0, value=float(settings.position_size_usdt), step=5.0
    )
    stop_loss = st.sidebar.number_input(
        "손절 (%)", min_value=0.1, value=float(settings.stop_loss_pct), step=0.1
    )
    atr_mult = st.sidebar.number_input(
        "Trailing ATR", min_value=0.5, value=float(settings.trailing_atr_mult), step=0.5
    )
    time_exit = st.sidebar.number_input(
        "시간청산 (h)", min_value=0.5, value=float(settings.time_exit_hours), step=0.5
    )

    settings.margin_mode = margin_mode
    settings.leverage = int(leverage)
    settings.position_size_usdt = float(notional)
    settings.stop_loss_pct = float(stop_loss)
    settings.trailing_atr_mult = float(atr_mult)
    settings.time_exit_hours = float(time_exit)
    STATE.update_settings(
        margin_mode=margin_mode, leverage=int(leverage), notional=float(notional),
        stop_loss_pct=float(stop_loss), trailing_atr_mult=float(atr_mult),
        time_exit_hours=float(time_exit),
    )

    st.sidebar.divider()
    col1, col2 = st.sidebar.columns(2)
    running = runner.is_alive() or STATE.running
    if col1.button(
        "▶ 시작", use_container_width=True, type="primary", disabled=running,
    ):
        mode = botmod.exchange_mode_label()
        STATE.set_running(True, status=f"시작 중 ({mode})")
        STATE.log("INFO", "system", "▶ 시작 버튼 — 봇 초기화 중 (FinBERT·거래소 연결)")
        runner.start()
        st.rerun()
    if col2.button("■ 정지", use_container_width=True, disabled=not running):
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


_CHART_HEIGHT = 210


def candlestick(symbol: str) -> go.Figure | None:
    ohlcv = STATE.get_ohlcv(symbol)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms")

    fig = go.Figure(
        data=[go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=symbol,
        )]
    )

    pos = next((p for p in STATE.get_positions() if p.symbol == symbol), None)
    if pos is not None:
        fig.add_hline(
            y=pos.stop_loss, line_dash="dash", line_color="#d62728",
            annotation_text=f"SL {pos.stop_loss:.2f}", annotation_position="right",
        )
        fig.add_hline(
            y=pos.trailing_stop, line_dash="dot", line_color="#2ca02c",
            annotation_text=f"TP {pos.trailing_stop:.2f}", annotation_position="right",
        )
        fig.add_hline(
            y=pos.entry_price, line_dash="solid", line_color="#1f77b4",
            annotation_text=f"Entry {pos.entry_price:.2f}", annotation_position="left",
        )

    fig.update_layout(
        height=_CHART_HEIGHT,
        margin=dict(l=4, r=4, t=22, b=4),
        xaxis_rangeslider_visible=False,
        title=dict(text=symbol, font=dict(size=11)),
        font=dict(size=10),
    )
    return fig


@st.fragment(run_every=2)
def render_dashboard() -> None:
    # ---- 실행 상태 배너 (시작 버튼 피드백) ----
    if runner.is_alive():
        if STATE.status.startswith("running"):
            st.success(f"🟢 **봇 실행 중** — {STATE.status}")
        elif STATE.status.startswith("시작"):
            st.warning(f"🟡 **봇 시작 중** — {STATE.status} · FinBERT/거래소 연결 중 (1~2분)")
        else:
            st.info(f"🔵 **봇 스레드 동작 중** — {STATE.status}")
    elif STATE.running:
        st.warning("🟡 **시작 요청됨** — 스레드 기동 대기 중…")
    else:
        st.info("⚪ **대기 중** — 사이드바 **▶ 시작** 버튼을 눌러 주세요.")

    positions = STATE.get_positions()

    c1, c2, c3 = st.columns(3)
    c1.metric("잔고 (USDT)", f"{STATE.get_balance():,.2f}")
    c2.metric("포지션", f"{len(positions)} / {settings.max_positions}")
    c3.metric("상태", STATE.status)

    st.markdown("##### 📌 오픈 포지션")
    if positions:
        df = pd.DataFrame([{
            "코인": p.symbol, "방향": p.side.upper(), "수량": round(p.amount, 4),
            "진입": round(p.entry_price, 2), "현재": round(p.mark_price, 2),
            "손절": round(p.stop_loss, 2), "익절": round(p.trailing_stop, 2),
            "PnL%": p.unrealized_pct,
        } for p in positions])
        st.dataframe(df, use_container_width=True, hide_index=True, height=70)
    else:
        st.caption("오픈 포지션 없음")

    st.markdown("##### 📈 차트 (15m)")
    open_syms = [p.symbol for p in positions]
    data_syms = STATE.symbols_with_data()
    chart_syms = (open_syms + [s for s in data_syms if s not in open_syms])[:2]
    if chart_syms:
        cols = st.columns(len(chart_syms))
        for col, sym in zip(cols, chart_syms):
            fig = candlestick(sym)
            if fig is not None:
                col.plotly_chart(fig, use_container_width=True, key=f"chart_{sym}")
    else:
        st.caption("차트 데이터 수집 중…")

    st.markdown("##### 📰 뉴스 · 로그")
    n_col, l_col = st.columns(2)

    with n_col:
        st.caption("실시간 뉴스 (EN + 한글)")
        news = STATE.get_news(30)
        with st.container(height=175):
            if not news:
                st.caption("뉴스 수집 대기 중…")
            for nw in news:
                icon = "🟢" if nw.score > 0.2 else "🔴" if nw.score < -0.2 else "⚪"
                ko = nw.title_ko or nw.title
                st.markdown(
                    f'<div class="news-item">{icon} <code>{nw.time}</code> '
                    f"<b>[{nw.score:+.2f} {nw.label}]</b><br>"
                    f"🇺🇸 {nw.title}<br>"
                    f'<span class="news-ko">🇰🇷 {ko}</span><br>'
                    f"<sub>{nw.source}</sub></div>",
                    unsafe_allow_html=True,
                )

    with l_col:
        st.caption("진입/청산 · 주문 실패")
        logs = STATE.get_logs(80)
        with st.container(height=175):
            if not logs:
                st.caption("로그 대기 중…")
            for lg in logs:
                color = {"ERROR": "🔴", "WARNING": "🟡"}.get(lg["level"], "🟢")
                st.markdown(
                    f'<div class="log-item">{color} <code>{lg["time"][-8:]}</code> '
                    f"<b>[{lg['category']}]</b> {lg['message']}</div>",
                    unsafe_allow_html=True,
                )


st.markdown("# 📊 바이낸스 USDⓈ-M 뉴스 트레이딩 봇")
render_sidebar()
render_dashboard()
