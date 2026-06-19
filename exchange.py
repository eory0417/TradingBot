"""ccxt를 통한 바이낸스 USDⓈ-M 선물 클라이언트 초기화.

비동기 ``ccxt.async_support.binance`` 클라이언트를 USDⓈ-M 무기한 선물 시장에
맞는 기본값, 테스트넷 지원, 표준화된 에러 처리/로깅으로 감싼다.

비동기 변형을 사용하는 이유는 이 트레이딩 시스템이 이벤트 기반(뉴스 수집 +
주문 제출)이며 I/O로 이벤트 루프를 막아서는 안 되기 때문이다.

사용 예
-------
    import asyncio
    from exchange import create_exchange, close_exchange

    async def main():
        ex = create_exchange()
        try:
            balance = await ex.fetch_balance()
            print(balance["USDT"]["free"])
        finally:
            await close_exchange(ex)

    asyncio.run(main())
"""

from __future__ import annotations

import ccxt.async_support as ccxt  # 비동기(논블로킹) ccxt 클라이언트

from config import settings
from logger import get_logger, log_exception

log = get_logger(__name__)


def create_exchange() -> ccxt.binance:
    """비동기 ccxt 바이낸스 USDⓈ-M 선물 클라이언트를 생성·설정한다.

    반환값
    ------
    ccxt.async_support.binance
        바로 사용 가능한 인증된 거래소 인스턴스. 종료 시 내부 aiohttp 세션을
        해제하기 위해 반드시 ``await close_exchange(ex)``(또는
        ``await ex.close()``)를 호출해야 한다.
    """
    exchange = ccxt.binance(
        {
            "apiKey": settings.binance_api_key.get_secret_value(),
            "secret": settings.binance_secret_key.get_secret_value(),
            "enableRateLimit": True,  # 바이낸스 레이트 리밋 준수
            "options": {
                # USDⓈ-M 무기한 선물(ccxt에서는 "future").
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "recvWindow": 10_000,
            },
        }
    )

    if settings.binance_testnet:
        # 모든 REST/WS 호출을 바이낸스 선물 테스트넷 엔드포인트로 라우팅한다.
        exchange.set_sandbox_mode(True)
        log.info("Binance client initialized in TESTNET (sandbox) mode")
    else:
        log.warning("Binance client initialized in LIVE (real funds) mode")

    return exchange


async def load_markets_safe(exchange: ccxt.binance) -> bool:
    """마켓을 로드하고 연결을 검증하며, 표준화된 에러 로깅을 수행한다.

    성공 시 ``True``, 네트워크/API 실패 시 ``False``를 반환한다(상세 원인은
    :func:`log_exception`으로 기록).
    """
    try:
        await exchange.load_markets()
        log.info(
            "Markets loaded successfully | symbols=%d",
            len(exchange.symbols or []),
        )
        return True
    except ccxt.AuthenticationError as exc:
        log_exception(log, exc, context="api_auth")
    except ccxt.NetworkError as exc:
        log_exception(log, exc, context="network")
    except ccxt.ExchangeError as exc:
        log_exception(log, exc, context="api_call")
    except Exception as exc:  # noqa: BLE001 - 최후의 안전망
        log_exception(log, exc, context="exchange_init")
    return False


async def close_exchange(exchange: ccxt.binance) -> None:
    """거래소의 내부 네트워크 세션을 안전하게 종료한다."""
    try:
        await exchange.close()
        log.debug("Exchange session closed")
    except Exception as exc:  # noqa: BLE001
        log_exception(log, exc, context="exchange_close")
