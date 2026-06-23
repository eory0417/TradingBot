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

import os
import sys
import time

import ccxt.async_support as ccxt  # 비동기(논블로킹) ccxt 클라이언트

from config import settings
from http_session import create_tcp_connector
from logger import format_exception_brief, format_exception_detail, get_logger, log_exception

log = get_logger(__name__)


def _patch_open_threaded_resolver(exchange: ccxt.binance) -> None:
    """aiodns(AsyncResolver)는 PyInstaller exe에서 DNS 실패가 잦다 → 시스템 DNS 사용."""
    import ssl

    def patched_open() -> None:
        if exchange.asyncio_loop is None:
            if sys.version_info >= (3, 7):
                exchange.asyncio_loop = __import__("asyncio").get_running_loop()
            else:
                exchange.asyncio_loop = __import__("asyncio").get_event_loop()
            exchange.throttler.loop = exchange.asyncio_loop

        if exchange.ssl_context is None:
            exchange.ssl_context = (
                ssl.create_default_context(cafile=exchange.cafile)
                if exchange.verify
                else exchange.verify
            )
            if exchange.ssl_context and exchange.safe_bool(
                exchange.options, "include_OS_certificates", False
            ):
                os_default_paths = ssl.get_default_verify_paths()
                if (
                    os_default_paths.cafile
                    and os_default_paths.cafile != exchange.cafile
                ):
                    exchange.ssl_context.load_verify_locations(
                        cafile=os_default_paths.cafile
                    )

        if exchange.own_session and exchange.session is None:
            import aiohttp

            exchange.tcp_connector = create_tcp_connector(
                loop=exchange.asyncio_loop,
                ssl=exchange.ssl_context,
            )
            exchange.session = aiohttp.ClientSession(
                loop=exchange.asyncio_loop,
                connector=exchange.tcp_connector,
                trust_env=exchange.aiohttp_trust_env,
            )

    exchange.open = patched_open  # type: ignore[method-assign]


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
                # spot sapi(currency config) 호출 생략 — 선물 전용 봇, 일부 네트워크에서
                # /sapi/v1/capital/config/getall 만 막히는 경우 완화.
                "fetchCurrencies": False,
            },
            "timeout": 30_000,
            "aiohttp_trust_env": True,
        }
    )

    _patch_open_threaded_resolver(exchange)

    if settings.binance_testnet:
        # 구 선물 테스트넷(sandbox)은 폐지됨 → Binance Demo Trading API 사용.
        # demo.binance.com 에서 발급한 API 키가 필요하다(실거래 키와 호환되지 않음).
        exchange.enable_demo_trading(True)
        log.info("Binance client initialized in DEMO trading mode")
    else:
        log.warning("Binance client initialized in LIVE (real funds) mode")

    return exchange


async def diagnose_exchange(exchange: ccxt.binance) -> list[str]:
    """Binance 공개/선물 API 연결 진단 — exe·SSL·네트워크 문제 추적용."""
    lines: list[str] = []
    mode = "DEMO" if settings.binance_testnet else "LIVE"
    lines.append(f"mode={mode} defaultType={exchange.options.get('defaultType')}")

    if getattr(sys, "frozen", False):
        lines.append(f"runtime=PyInstaller exe={sys.executable}")
    else:
        lines.append(f"runtime=python exe={sys.executable}")

    lines.append(f"SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE', '(unset)')}")
    try:
        import certifi

        ca = certifi.where()
        lines.append(f"certifi={ca} exists={os.path.isfile(ca)}")
        if getattr(sys, "frozen", False):
            bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "certifi", "cacert.pem")
            lines.append(f"certifi_bundled={bundled} exists={os.path.isfile(bundled)}")
    except ImportError:
        lines.append("certifi=NOT_INSTALLED")

    import socket

    for host in ("api.binance.com", "fapi.binance.com"):
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            lines.append(f"dns_socket={host} OK")
        except OSError as exc:
            lines.append(f"dns_socket={host} FAIL | {exc}")

    lines.append("dns_resolver=ThreadedResolver (aiodns 우회)")

    def _resolve(*names: str):
        for name in names:
            fn = getattr(exchange, name, None)
            if callable(fn):
                return fn
        return None

    # ccxt snake_case: fapipublic_get_ping (not fapi_public_get_ping)
    tests: list[tuple[str, tuple[str, ...]]] = [
        ("spot_ping", ("public_get_ping", "publicGetPing")),
        ("spot_time", ("public_get_time", "publicGetTime")),
        ("futures_ping", ("fapipublic_get_ping", "fapiPublicGetPing")),
        ("futures_time", ("fapipublic_get_time", "fapiPublicGetTime")),
    ]
    for name, method_names in tests:
        fn = _resolve(*method_names)
        if fn is None:
            lines.append(f"{name}=SKIP (no method: {method_names[0]})")
            continue
        t0 = time.monotonic()
        try:
            await fn()
            ms = (time.monotonic() - t0) * 1000
            lines.append(f"{name}=OK ({ms:.0f}ms)")
        except Exception as exc:  # noqa: BLE001
            ms = (time.monotonic() - t0) * 1000
            lines.append(f"{name}=FAIL ({ms:.0f}ms) | {format_exception_brief(exc)}")
            log_exception(log, exc, context=f"diagnose_{name}")

    return lines


async def load_markets_safe(exchange: ccxt.binance) -> tuple[bool, str | None]:
    """마켓을 로드하고 연결을 검증하며, 표준화된 에러 로깅을 수행한다.

    성공 시 (True, None), 실패 시 (False, 상세 오류 문자열).
    """
    last_exc: BaseException | None = None
    try:
        await exchange.load_markets()
        log.info(
            "Markets loaded successfully | symbols=%d",
            len(exchange.symbols or []),
        )
        return True, None
    except ccxt.AuthenticationError as exc:
        last_exc = exc
        log_exception(log, exc, context="api_auth")
    except ccxt.NetworkError as exc:
        last_exc = exc
        log_exception(log, exc, context="network")
    except ccxt.ExchangeError as exc:
        last_exc = exc
        log_exception(log, exc, context="api_call")
    except Exception as exc:  # noqa: BLE001 - 최후의 안전망
        last_exc = exc
        log_exception(log, exc, context="exchange_init")
    if last_exc is not None:
        return False, format_exception_detail(last_exc)
    return False, "unknown error"


async def close_exchange(exchange: ccxt.binance) -> None:
    """거래소의 내부 네트워크 세션을 안전하게 종료한다."""
    try:
        await exchange.close()
        log.debug("Exchange session closed")
    except Exception as exc:  # noqa: BLE001
        log_exception(log, exc, context="exchange_close")
