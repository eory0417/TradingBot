"""1단계 헬스 체크 / 스모크 테스트.

기반 모듈들이 올바르게 연결되는지 검증한다.

  1. ``.env``에서 설정이 로드·검증되는지.
  2. 로깅 시스템이 콘솔 + 롤링 파일에 기록하는지.
  3. ccxt 바이낸스 USDⓈ-M 클라이언트가 초기화되는지(실제 자격증명이 있으면
     API까지 도달).
  4. 텔레그램 알림기가 초기화되는지(실제 자격증명이 있으면 테스트 메시지 전송).

실행:  python healthcheck.py
플레이스홀더 자격증명이면 네트워크 단계는 *우아하게* 실패하고 표준 형식으로
기록된다 — 이것 자체가 에러 처리 동작을 보여준다.
"""

from __future__ import annotations

import asyncio

from config import settings
from exchange import close_exchange, create_exchange, load_markets_safe
from logger import get_logger
from notifier import TelegramNotifier

log = get_logger("healthcheck")


def _has_real_credentials() -> bool:
    """휴리스틱: 자격증명이 실제처럼 보이는지(번들된 플레이스홀더가 아닌지)."""
    api_key = settings.binance_api_key.get_secret_value()
    return bool(api_key) and "your_" not in api_key


async def main() -> None:
    log.info("=== Stage-1 health check started ===")
    log.info(
        "Config OK | testnet=%s | log_level=%s | log_dir=%s",
        settings.binance_testnet,
        settings.log_level,
        settings.log_path,
    )

    # ---- 거래소 ----
    exchange = create_exchange()
    try:
        if _has_real_credentials():
            ok, market_err = await load_markets_safe(exchange)
            log.info("Exchange load_markets: %s", "OK" if ok else "FAILED")
            if market_err:
                log.error("load_markets detail:\n%s", market_err)
            try:
                await exchange.publicGetPing()
                log.info("Binance public ping: OK")
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Binance public ping FAILED (%s: %s) — "
                    "방화벽·IP 화이트리스트·api.binance.com 차단 여부 확인",
                    type(exc).__name__,
                    exc,
                )
        else:
            log.warning(
                "Skipping live exchange call: placeholder credentials detected. "
                "Client object initialized successfully."
            )
    finally:
        await close_exchange(exchange)

    # ---- 텔레그램 ----
    notifier = TelegramNotifier()
    try:
        if _has_real_credentials():
            sent = await notifier.send("✅ Stage-1 health check: notifier online")
            log.info("Telegram send: %s", "OK" if sent else "FAILED")
        else:
            log.warning(
                "Skipping live Telegram send: placeholder credentials detected. "
                "Notifier object initialized successfully."
            )
    finally:
        await notifier.close()

    # ---- coinnesskr 수신(선택) ----
    log.info("News source mode: %s", settings.news_source_mode)
    if settings.use_coinnesskr:
        await _check_coinnesskr()

    log.info("=== Stage-1 health check finished ===")


async def _check_coinnesskr() -> None:
    """coinnesskr 모드일 때 세션/채널 접근 가능 여부를 스모크 테스트한다."""
    from telegram_news import CoinnessListener

    if not settings.telegram_api_id or not settings.telegram_api_hash_value:
        log.warning("coinnesskr: TELEGRAM_API_ID/HASH 미설정 — 수신 비활성")
        return
    if not settings.telegram_session_path.exists():
        log.warning(
            "coinnesskr: 세션 파일 없음 (%s) — `python telegram_login.py` 실행 필요",
            settings.telegram_session_path,
        )
        return

    listener = CoinnessListener()
    ok = await listener.connect()
    if not ok:
        log.error("coinnesskr: 연결/인증 실패 — telegram_login.py 재실행 필요")
        return
    try:
        recent = await listener.warmup_recent(3)
        log.info("coinnesskr: 연결 OK | 최근 메시지 %d건 파싱", len(recent))
    finally:
        await listener.stop()


if __name__ == "__main__":
    asyncio.run(main())
