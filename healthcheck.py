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
            ok = await load_markets_safe(exchange)
            log.info("Exchange connectivity: %s", "OK" if ok else "FAILED")
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

    log.info("=== Stage-1 health check finished ===")


if __name__ == "__main__":
    asyncio.run(main())
