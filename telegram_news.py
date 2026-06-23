"""Telethon 기반 coinness 텔레그램 채널 실시간 수신 (Plus 전용 뉴스 소스).

알림 발송 봇(:mod:`notifier` / ``TELEGRAM_TOKEN``)과는 완전히 별개로, 개인 유저
세션(``TELEGRAM_API_ID`` / ``TELEGRAM_API_HASH``)으로 채널 메시지를 수신한다.
세션 파일(``models/<name>.session``)은 :mod:`telegram_login` 으로 1회 생성해야 한다.

채널은 ``COINNESS_CHANNEL`` 로 지정하며 ``COINNESS_LANG`` 으로 언어를 구분한다.

  * ``ko`` (coinnesskr): 한국어 헤드라인 → ``origin="coinnesskr"``, ``title_ko`` 에
    원문 보관 후 공통 파이프라인에서 KO→EN 번역하여 FinBERT 점수화.
  * ``en`` (coinnessGL): 영어 헤드라인 → ``origin="coinnessgl"``, 번역 없이 원문
    그대로 FinBERT 점수화(GUI 한글 표시는 EN→KO 번역으로 처리).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Awaitable, Callable

from config import settings
from logger import get_logger, log_exception
from news_analyzer import NewsItem

log = get_logger(__name__)

# "[제목]본문 ... https://coinness.com/news/123" 형식 파싱용.
_TITLE_RE = re.compile(r"^\s*\[([^\]]+)\]")
_URL_RE = re.compile(r"https?://\S+")
_COINNESS_URL_RE = re.compile(r"https?://(?:www\.)?coinness\.com/\S+", re.IGNORECASE)
_MAX_TITLE_CHARS = 300

CoinnessCallback = Callable[[NewsItem], Awaitable[None]]


def parse_coinness_message(message) -> NewsItem | None:
    """Telethon 메시지를 :class:`NewsItem` 으로 변환한다(파싱 불가 시 ``None``).

    제목은 선행 ``[...]`` 안 텍스트(없으면 본문 첫 줄)에서 추출하고, 본문에서
    URL을 제거한다. coinness.com 링크가 있으면 ``url`` 로, 없으면 안정 ID만 사용.
    ``COINNESS_LANG`` 에 따라 한국어/영어 채널을 구분해 origin·title 을 채운다.
    """
    text = (getattr(message, "message", "") or "").strip()
    if not text:
        return None

    url_match = _COINNESS_URL_RE.search(text)
    url = url_match.group(0) if url_match else ""

    title_match = _TITLE_RE.search(text)
    if title_match:
        headline = title_match.group(1).strip()
    else:
        first_line = _URL_RE.sub("", text).strip().splitlines()
        headline = first_line[0].strip() if first_line else ""
    headline = headline[:_MAX_TITLE_CHARS].strip()
    if not headline:
        return None

    msg_id = getattr(message, "id", 0)
    chat_id = getattr(message, "chat_id", None) or settings.coinness_channel
    published_at = getattr(message, "date", None)
    if not isinstance(published_at, datetime):
        published_at = datetime.now(timezone.utc)
    elif published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    is_english = settings.coinness_is_english
    return NewsItem(
        id=url or f"tg:{chat_id}:{msg_id}",
        # 한국어 채널은 파이프라인에서 EN 번역본으로 title 을 덮어쓴다.
        # 영어 채널은 원문이 이미 영어이므로 그대로 분석 입력으로 쓴다.
        title=headline,
        url=url,
        source=settings.coinness_channel,
        published_at=published_at,
        origin="coinnessgl" if is_english else "coinnesskr",
        # 영어 채널은 한글 원문이 없으므로 비워 둔다(GUI 는 EN→KO 번역으로 표시).
        title_ko="" if is_english else headline,
    )


class CoinnessListener:
    """``COINNESS_CHANNEL`` 채널을 Telethon 유저 세션으로 수신한다.

    수명주기: :meth:`connect` → (:meth:`warmup_recent`) → :meth:`add_handler`
    → ... → :meth:`stop`. 자격증명/세션이 없으면 :meth:`connect` 가 ``False`` 를
    반환하므로 호출 측은 우아하게 건너뛸 수 있다.
    """

    def __init__(self) -> None:
        self._client = None
        self._channel = settings.coinness_channel

    def _credentials_ok(self) -> bool:
        return bool(settings.telegram_api_id) and bool(settings.telegram_api_hash_value)

    async def connect(self) -> bool:
        """클라이언트를 연결하고 세션 인증 여부를 확인한다(성공 시 ``True``)."""
        if not self._credentials_ok():
            log.warning(
                "coinness 비활성: TELEGRAM_API_ID/HASH 미설정 — .env 를 확인하세요."
            )
            return False
        session_path = settings.telegram_session_path
        if not session_path.exists():
            log.warning(
                "coinness 세션 없음 (%s) — 먼저 `python telegram_login.py` 를 실행하세요.",
                session_path,
            )
            return False

        try:
            from telethon import TelegramClient
        except ImportError:
            log.warning("coinness 비활성: telethon 미설치 (pip install telethon)")
            return False

        # Telethon은 세션명에 .session 을 자동으로 붙이므로 확장자를 제외한다.
        session_name = str(session_path.with_suffix(""))
        self._client = TelegramClient(
            session_name,
            settings.telegram_api_id,
            settings.telegram_api_hash_value,
        )
        try:
            await self._client.connect()
            if not await self._client.is_user_authorized():
                log.warning(
                    "coinness 세션 미인증 — `python telegram_login.py` 를 다시 실행하세요."
                )
                await self._client.disconnect()
                self._client = None
                return False
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="coinness_connect")
            self._client = None
            return False

        log.info("coinness 연결됨 | channel=@%s", self._channel)
        return True

    async def warmup_recent(self, limit: int) -> list[NewsItem]:
        """최근 메시지 ``limit`` 건을 가져와 :class:`NewsItem` 목록으로 반환한다."""
        if self._client is None:
            return []
        try:
            messages = await self._client.get_messages(self._channel, limit=limit)
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="coinness_warmup")
            return []
        items: list[NewsItem] = []
        for msg in messages:
            item = parse_coinness_message(msg)
            if item is not None:
                items.append(item)
        return items

    def add_handler(self, on_item: CoinnessCallback) -> None:
        """새 메시지 수신 핸들러를 등록한다(논블로킹)."""
        if self._client is None:
            return
        from telethon import events

        async def _handler(event) -> None:
            try:
                item = parse_coinness_message(event.message)
                if item is not None:
                    await on_item(item)
            except Exception as exc:  # noqa: BLE001 - 핸들러가 루프를 죽이면 안 됨
                log_exception(log, exc, context="coinness_handler")

        self._client.add_event_handler(
            _handler, events.NewMessage(chats=self._channel)
        )

    async def stop(self) -> None:
        """클라이언트 연결을 해제한다(중복 호출 안전)."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as exc:  # noqa: BLE001
                log.debug("coinness disconnect skipped | %s", exc)
            finally:
                self._client = None
