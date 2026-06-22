"""coinnesskr 수신용 Telethon 세션을 1회 생성하는 대화형 로그인 스크립트.

실행:  python telegram_login.py

전화번호 → 인증코드 → (2FA 비밀번호) 를 입력하면 ``models/<name>.session`` 파일이
생성된다. 이후 봇은 이 세션으로 @coinnesskr 메시지를 자동 수신한다.

필요 설정(.env): TELEGRAM_API_ID, TELEGRAM_API_HASH, (선택) TELEGRAM_SESSION_NAME.
API_ID/HASH 는 https://my.telegram.org 에서 발급한다(알림 봇 토큰과 무관).
"""

from __future__ import annotations

import asyncio
import sys

from config import settings


async def main() -> None:
    if not settings.telegram_api_id or not settings.telegram_api_hash_value:
        print(
            "오류: TELEGRAM_API_ID / TELEGRAM_API_HASH 가 설정되지 않았습니다.\n"
            "      https://my.telegram.org 에서 발급해 .env 에 입력하세요."
        )
        sys.exit(1)

    try:
        from telethon import TelegramClient
    except ImportError:
        print("오류: telethon 이 설치되어 있지 않습니다. 'pip install telethon' 후 다시 실행하세요.")
        sys.exit(1)

    session_path = settings.telegram_session_path
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_name = str(session_path.with_suffix(""))

    print(f"세션 파일 위치: {session_path}")
    print(f"수신 대상 채널: @{settings.coinness_channel}")

    client = TelegramClient(
        session_name, settings.telegram_api_id, settings.telegram_api_hash_value
    )
    # start() 는 미인증 시 전화번호/코드/2FA 를 대화형으로 요청한다.
    await client.start()

    me = await client.get_me()
    username = getattr(me, "username", None) or getattr(me, "first_name", "user")
    print(f"로그인 성공: {username}")

    try:
        entity = await client.get_entity(settings.coinness_channel)
        title = getattr(entity, "title", settings.coinness_channel)
        print(f"채널 접근 확인: {title}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"경고: 채널 @{settings.coinness_channel} 접근 실패 ({exc}). "
            "텔레그램에서 해당 채널을 먼저 구독했는지 확인하세요."
        )

    await client.disconnect()
    print(f"완료. 세션이 저장되었습니다: {session_path}")


if __name__ == "__main__":
    asyncio.run(main())
