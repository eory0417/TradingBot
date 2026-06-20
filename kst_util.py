"""UI 표시용 한국 표준시(KST, UTC+9) 변환.

Streamlit 내장 ``time_util`` 과 이름 충돌을 피하기 위해 ``kst_util`` 로 분리.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

KST = timezone(timedelta(hours=9))
TZ_LABEL = "KST"

__all__ = [
    "KST",
    "TZ_LABEL",
    "format_gui_hms",
    "format_kst",
    "format_legacy_stored_hms",
    "format_ms_kst",
    "ms_to_kst_pandas",
    "now_kst",
    "series_ms_to_kst_pandas",
    "to_kst",
]


def now_kst() -> datetime:
    return datetime.now(KST)


def to_kst(dt: datetime) -> datetime:
    """aware/naive datetime 을 KST 로 변환(naive 는 UTC 로 간주)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)


def format_kst(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return to_kst(dt).strftime(fmt)


def format_ms_kst(ms: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if not ms:
        return ""
    return format_kst(datetime.fromtimestamp(ms / 1000, tz=timezone.utc), fmt)


def format_legacy_stored_hms(stored: str) -> str:
    """저장된 시각 문자열을 KST HH:MM:SS 로 (legacy: naive = UTC)."""
    stored = (stored or "").strip()
    if not stored:
        return ""
    if len(stored) <= 8 and stored.count(":") >= 2 and " " not in stored:
        return stored[-8:] if len(stored) > 8 else stored
    try:
        dt = datetime.fromisoformat(stored.replace("Z", "+00:00"))
    except ValueError:
        return stored[-8:] if len(stored) >= 8 else stored
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return format_kst(dt, "%H:%M:%S")


def format_gui_hms(*, at_ms: int = 0, stored: str = "") -> str:
    """GUI용 시:분:초 — epoch ms 우선, 없으면 legacy 문자열(UTC naive)을 KST 로."""
    if at_ms:
        return format_ms_kst(at_ms, "%H:%M:%S")
    return format_legacy_stored_hms(stored)


def ms_to_kst_pandas(ms: int) -> pd.Timestamp:
    """Plotly 축/마커용 KST 타임스탬프(naive — 축 라벨이 KST 로 읽히게)."""
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert(KST).tz_localize(None)


def series_ms_to_kst_pandas(series: pd.Series) -> pd.Series:
    """OHLCV epoch ms 열 → KST naive datetime 열."""
    return pd.to_datetime(series, unit="ms", utc=True).dt.tz_convert(KST).dt.tz_localize(None)
