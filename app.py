"""Streamlit 웹 GUI 대시보드 (4단계).

원격 서버 배포에 유리한 단일 프로세스 구성으로, 트레이딩 봇(:mod:`bot`)을
백그라운드 스레드에서 실행하고 공유 상태(:mod:`state`)를 실시간으로 시각화한다.

화면 구성
---------
  * 상단    : 계정 잔고(USDT), 실행 상태, 오픈 포지션 현황.
  * 사이드바 : 설정 입력 패널(증거금 모드/레버리지/투자금/손절% 등) + 시작/정지.
  * 중앙    : 진입 코인의 가격 흐름 + 동적 익절/손절 라인 캔들 차트 2개.
  * 하단    : 실시간 뉴스/감성 점수, 진입·청산 로그, 주문 실패 사유 스크롤 출력.

실행:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bot as botmod
from config import settings
from state import STATE

st.set_page_config(page_title="뉴스 트레이딩 봇", layout="wide", page_icon="📊")


# --------------------------------------------------------------------------- #
#  백그라운드 봇 러너(프로세스 전역 싱글톤)
# --------------------------------------------------------------------------- #
class BotRunner:
    """봇을 자체 이벤트 루프를 가진 데몬 스레드에서 구동한다."""

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.bot: botmod.TradingBot | None = None

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.is_alive():
            return
        self.bot = botmod.TradingBot()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.bot.run())

        self.thread = threading.Thread(target=_run, daemon=True, name="trading-bot")
        self.thread.start()

    def stop(self) -> None:
        if self.bot is not None:
            self.bot.stop()


@st.cache_resource
def get_runner() -> BotRunner:
    return BotRunner()


runner = get_runner()


# --------------------------------------------------------------------------- #
#  사이드바: 설정 입력 패널 + 제어
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    st.sidebar.header("⚙️ 설정")

    margin_mode = st.sidebar.selectbox(
        "증거금 모드", ["isolated", "cross"],
        index=0 if settings.margin_mode == "isolated" else 1,
    )
    leverage = st.sidebar.slider("레버리지 배수", 1, 25, settings.leverage)
    notional = st.sidebar.number_input(
        "투자금 / 진입 (USDT)", min_value=5.0, value=float(settings.position_size_usdt), step=5.0
    )
    stop_loss = st.sidebar.number_input(
        "고정 손절 (%)", min_value=0.1, value=float(settings.stop_loss_pct), step=0.1
    )
    atr_mult = st.sidebar.number_input(
        "Trailing ATR 배수(기본)", min_value=0.5, value=float(settings.trailing_atr_mult), step=0.5
    )
    time_exit = st.sidebar.number_input(
        "시간 청산 (시간)", min_value=0.5, value=float(settings.time_exit_hours), step=0.5
    )

    # 설정을 즉시 반영(다음 진입부터 적용).
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
    if col1.button("▶ 시작", use_container_width=True, type="primary"):
        runner.start()
        st.toast("봇을 시작했습니다.")
    if col2.button("■ 정지", use_container_width=True):
        runner.stop()
        st.toast("봇 정지 요청을 보냈습니다.")

    mode = "SIM (페이퍼)" if not botmod.has_real_credentials() else "LIVE"
    st.sidebar.caption(f"모드: **{mode}** · 상태: **{STATE.status}**")


# --------------------------------------------------------------------------- #
#  차트
# --------------------------------------------------------------------------- #
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

    # 현재 포지션의 동적 익절/손절 라인 오버레이.
    pos = next((p for p in STATE.get_positions() if p.symbol == symbol), None)
    if pos is not None:
        fig.add_hline(
            y=pos.stop_loss, line_dash="dash", line_color="#d62728",
            annotation_text=f"손절 {pos.stop_loss:.4f}", annotation_position="right",
        )
        fig.add_hline(
            y=pos.trailing_stop, line_dash="dot", line_color="#2ca02c",
            annotation_text=f"익절(Trailing x{pos.atr_mult}) {pos.trailing_stop:.4f}",
            annotation_position="right",
        )
        fig.add_hline(
            y=pos.entry_price, line_dash="solid", line_color="#1f77b4",
            annotation_text=f"진입 {pos.entry_price:.4f}", annotation_position="left",
        )

    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, title=symbol,
    )
    return fig


# --------------------------------------------------------------------------- #
#  실시간 대시보드(프래그먼트: 일정 주기로 자동 갱신)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=2)
def render_dashboard() -> None:
    # ---- 상단 지표 ----
    positions = STATE.get_positions()
    c1, c2, c3 = st.columns(3)
    c1.metric("계정 잔고 (USDT)", f"{STATE.get_balance():,.2f}")
    c2.metric("오픈 포지션", f"{len(positions)} / {settings.max_positions}")
    c3.metric("상태", STATE.status)

    # ---- 오픈 포지션 현황 ----
    st.subheader("📌 오픈 포지션")
    if positions:
        df = pd.DataFrame([{
            "코인": p.symbol, "방향": p.side.upper(), "수량": round(p.amount, 6),
            "진입가": round(p.entry_price, 4), "현재가": round(p.mark_price, 4),
            "손절가": round(p.stop_loss, 4), "익절(Trailing)": round(p.trailing_stop, 4),
            "ATR배수": p.atr_mult, "미실현%": p.unrealized_pct,
            "진입시각": p.opened_at,
        } for p in positions])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("현재 오픈된 포지션이 없습니다.")

    # ---- 캔들 차트 2개 ----
    st.subheader("📈 가격 & 동적 익절/손절 라인")
    open_syms = [p.symbol for p in positions]
    data_syms = STATE.symbols_with_data()
    # 우선 오픈 포지션 코인, 부족하면 데이터 있는 코인으로 2개 채움.
    chart_syms = (open_syms + [s for s in data_syms if s not in open_syms])[:2]
    if chart_syms:
        cols = st.columns(len(chart_syms))
        for col, sym in zip(cols, chart_syms):
            fig = candlestick(sym)
            if fig is not None:
                col.plotly_chart(fig, use_container_width=True, key=f"chart_{sym}")
    else:
        st.info("차트 데이터를 수집 중입니다...")

    # ---- 하단: 뉴스 / 로그 ----
    st.subheader("📰 실시간 뉴스 · 로그")
    n_col, l_col = st.columns(2)

    with n_col:
        st.caption("실시간 뉴스 & 감성 점수")
        news = STATE.get_news(40)
        with st.container(height=260):
            if not news:
                st.write("뉴스 수집 대기 중...")
            for nw in news:
                icon = "🟢" if nw.score > 0.2 else "🔴" if nw.score < -0.2 else "⚪"
                st.markdown(
                    f"{icon} `{nw.time}` **[{nw.score:+.2f} {nw.label}]** "
                    f"{nw.title}  \n<sub>{nw.source}</sub>",
                    unsafe_allow_html=True,
                )

    with l_col:
        st.caption("진입/청산 로그 · 주문 실패 사유")
        logs = STATE.get_logs(120)
        with st.container(height=260):
            if not logs:
                st.write("로그 대기 중...")
            for lg in logs:
                color = {"ERROR": "🔴", "WARNING": "🟡"}.get(lg["level"], "🟢")
                st.markdown(
                    f"{color} `{lg['time']}` **[{lg['category']}]** {lg['message']}"
                )


# --------------------------------------------------------------------------- #
#  메인
# --------------------------------------------------------------------------- #
st.title("📊 바이낸스 USDⓈ-M 뉴스 트레이딩 봇")
render_sidebar()
render_dashboard()
