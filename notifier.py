"""비동기 텔레그램 알림 모듈.

운영 알림(진입/청산, 에러, 시스템 상태)을 보내기 위해
``python-telegram-bot``(가장 널리 쓰이는 프로덕션 등급 텔레그램 라이브러리)을
얇게 감싼 견고한 비동기 래퍼 :class:`TelegramNotifier`를 제공한다.

모든 네트워크/API 실패는 표준화된 로깅 헬퍼를 통해 포착·기록되므로, 알림 전송
실패가 트레이딩 루프를 중단시키지 않는다.

사용 예
-------
    import asyncio
    from notifier import TelegramNotifier

    async def main():
        notifier = TelegramNotifier()
        await notifier.send("🚀 Bot online")
        await notifier.send_error("entry_order", "BTC/USDT order rejected")
        await notifier.close()

    asyncio.run(main())
"""

from __future__ import annotations

import html
from typing import Final

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from config import settings
from logger import get_logger, log_exception

log = get_logger(__name__)


class TelegramNotifier:
    """봇 알림용 비동기 텔레그램 메시지 발송기.

    단일 인스턴스를 프로세스 생애 주기 동안 재사용할 수 있다. 내부 HTTP 커넥션
    풀은 ``python-telegram-bot``이 관리하며, 종료 시 :meth:`close`를 호출한다.
    """

    def __init__(self, chat_id: str | None = None) -> None:
        self._token: Final[str] = settings.telegram_token.get_secret_value()
        self._chat_id: Final[str] = chat_id or settings.telegram_chat_id
        self._bot: Bot = Bot(token=self._token)

    async def send(self, message: str, *, parse_mode: str = ParseMode.HTML) -> bool:
        """설정된 채팅으로 메시지를 전송한다.

        성공 시 ``True``, 실패 시 ``False``를 반환한다(실패 원인은 기록하되 예외를
        다시 던지지 않으므로 트레이딩 루프 내 호출자는 안전하다).
        """
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
            log.debug("Telegram message sent | chars=%d", len(message))
            return True
        except TelegramError as exc:
            log_exception(log, exc, context="telegram_send", chat_id=self._chat_id)
        except Exception as exc:  # noqa: BLE001 - 알림 실패가 호출자를 죽이지 않게 함
            log_exception(log, exc, context="telegram_send", chat_id=self._chat_id)
        return False

    async def send_error(self, context: str, detail: str) -> bool:
        """표준화되고 이스케이프 처리된 에러 알림을 전송한다."""
        text = (
            "⚠️ <b>ERROR</b>\n"
            f"<b>Context:</b> {html.escape(context)}\n"
            f"<b>Detail:</b> {html.escape(detail)}"
        )
        return await self.send(text)

    async def send_trade(self, action: str, symbol: str, detail: str = "") -> bool:
        """표준화된 거래 이벤트 알림을 전송한다."""
        text = (
            "📈 <b>TRADE</b>\n"
            f"<b>Action:</b> {html.escape(action)}\n"
            f"<b>Symbol:</b> {html.escape(symbol)}"
        )
        if detail:
            text += f"\n<b>Info:</b> {html.escape(detail)}"
        return await self.send(text)

    async def send_position_open(
        self,
        *,
        symbol: str,
        side: str,
        amount_usdt: float,
        entry_price: float,
        news: str,
        score: float,
        news_ko: str = "",
        leverage: int | None = None,
    ) -> bool:
        """포지션 오픈 알림.

        요구 양식: [뉴스 내용(영문+한글) / 뉴스 점수 / 포지션 금액 / 코인명 / 진입가]를 포함.
        """
        arrow = "🟢 LONG" if side.lower() == "long" else "🔴 SHORT"
        lev_line = f"<b>레버리지:</b> {leverage}x\n" if leverage else ""
        text = (
            f"🚀 <b>포지션 오픈</b> {arrow}\n"
            f"<b>코인명:</b> {html.escape(symbol)}\n"
            f"<b>진입가:</b> {entry_price:,.4f}\n"
            f"<b>포지션 금액:</b> {amount_usdt:,.2f} USDT\n"
            f"{lev_line}"
            f"<b>뉴스 점수:</b> {score:+.3f}\n"
            f"<b>뉴스(EN):</b> {html.escape(news or 'N/A')}\n"
            f"<b>뉴스(한글):</b> {html.escape(news_ko or '번역 없음')}"
        )
        return await self.send(text)

    async def send_position_close(
        self,
        *,
        symbol: str,
        side: str,
        amount_usdt: float,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        reason: str,
        news: str,
        score: float,
        pnl_usdt: float | None = None,
        news_ko: str = "",
    ) -> bool:
        """포지션 청산 알림.

        요구 양식: [뉴스 내용(영문+한글) / 뉴스 점수 / 손익(% + 금액) / 포지션 금액 /
        코인명 / 진입·청산가]를 포함.
        """
        result_icon = "✅" if pnl_pct >= 0 else "❌"
        side_label = "LONG" if side.lower() == "long" else "SHORT"
        if pnl_usdt is not None:
            pnl_line = f"<b>손익:</b> {pnl_pct:+.2f}% ({pnl_usdt:+,.2f} USDT)\n"
        else:
            pnl_line = f"<b>손익:</b> {pnl_pct:+.2f}%\n"
        text = (
            f"🏁 <b>포지션 청산</b> {result_icon} ({side_label})\n"
            f"<b>코인명:</b> {html.escape(symbol)}\n"
            f"<b>진입가:</b> {entry_price:,.4f}\n"
            f"<b>청산가:</b> {exit_price:,.4f}\n"
            f"{pnl_line}"
            f"<b>포지션 금액:</b> {amount_usdt:,.2f} USDT\n"
            f"<b>청산 사유:</b> {html.escape(reason)}\n"
            f"<b>뉴스 점수:</b> {score:+.3f}\n"
            f"<b>뉴스(EN):</b> {html.escape(news or 'N/A')}\n"
            f"<b>뉴스(한글):</b> {html.escape(news_ko or '번역 없음')}"
        )
        return await self.send(text)

    async def close(self) -> None:
        """봇의 내부 네트워크 리소스를 해제한다."""
        try:
            await self._bot.shutdown()
            log.debug("Telegram bot shut down")
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="telegram_close")
