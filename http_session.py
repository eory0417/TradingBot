"""aiohttp 커넥터·세션 공통 팩토리.

RSS 뉴스 수집(``news_analyzer``)과 ccxt 거래소(``exchange``)가 HTTP 요청 시
동일한 DNS/커넥터 설정을 쓴다. PyInstaller exe 환경에서 aiodns 가 실패하는
경우가 많아 ``ThreadedResolver``(시스템 DNS)를 공통으로 사용한다.
"""

from __future__ import annotations

import aiohttp
from aiohttp.resolver import ThreadedResolver

# RSS/CryptoPanic 수집용 기본 헤더.
DEFAULT_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": "NewsTradingBot/1.0 (+https://example.local)",
}
DEFAULT_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)


def create_tcp_connector(
    *,
    loop=None,
    ssl=None,
) -> aiohttp.TCPConnector:
    """ThreadedResolver 기반 TCP 커넥터 (ccxt·RSS 공용)."""
    kwargs: dict = {
        "resolver": ThreadedResolver(),
        "enable_cleanup_closed": True,
    }
    if loop is not None:
        kwargs["loop"] = loop
    if ssl is not None:
        kwargs["ssl"] = ssl
    return aiohttp.TCPConnector(**kwargs)


def make_client_session(
    *,
    headers: dict[str, str] | None = None,
    trust_env: bool = True,
    loop=None,
) -> aiohttp.ClientSession:
    """RSS/CryptoPanic 폴링용 aiohttp 세션."""
    return aiohttp.ClientSession(
        headers=headers or DEFAULT_HTTP_HEADERS,
        connector=create_tcp_connector(loop=loop),
        trust_env=trust_env,
    )
