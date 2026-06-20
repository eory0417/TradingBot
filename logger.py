"""중앙 집중식 프로덕션 등급 로깅 모듈.

애플리케이션 전반에서 사용하는 단일 설정 로거 팩토리를 제공하여, 모든 모듈이
하나의 표준 형식으로 다음 두 곳에 동시에 로그를 출력하도록 한다.

  * 콘솔(stdout), 그리고
  * 설정된 ``logs/`` 디렉터리 내 롤링(rotating) 파일.

또한 진입 실패, 네트워크 에러, API 호출 오류에 대해 표준화된 실패 레코드를
남기는 헬퍼 :func:`log_exception`을 제공한다. 이 헬퍼는 상세 원인(에러 타입,
에러 코드, 메시지)과 전체 트레이스백을 파일 로그에 기록한다.

사용 예
-------
    from logger import get_logger, log_exception

    log = get_logger(__name__)
    log.info("Bot started")

    try:
        ...
    except Exception as exc:
        log_exception(log, exc, context="entry_order", symbol="BTC/USDT")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

from config import settings

# 콘솔과 파일 핸들러가 공유하는 표준 로그 라인 형식.
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | "
    "%(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 롤링 파일 핸들러 한도(파일당 10MB, 백업 5개 유지).
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

# 루트 로거에 핸들러가 정확히 한 번만 부착되도록 하는 가드.
_CONFIGURED = False


def _configure_root() -> None:
    """루트 로거에 콘솔 + 롤링 파일 핸들러를 한 번만 부착한다."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level, logging.INFO)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Streamlit은 stdout을 래핑하므로 Windows에서 reconfigure 시 OSError가 난다.
    # Streamlit 실행 시에는 파일 로그만 사용하고, 터미널 직접 실행 시에만 콘솔 출력.
    in_streamlit = "streamlit" in sys.modules
    if not in_streamlit:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                if sys.stdout.isatty():
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except Exception:
                pass
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

    # ---- 롤링 파일 핸들러 ----
    log_dir = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_dir / "trading_bot.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # 시끄러운 서드파티 라이브러리 로그 레벨을 낮춘다.
    for noisy in ("httpx", "httpcore", "telegram", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """전역 설정을 공유하는 모듈 단위 로거를 반환한다."""
    _configure_root()
    return logging.getLogger(name)


def _extract_error_code(exc: BaseException) -> str:
    """일반적인 예외에서 에러/상태 코드를 최대한 추출한다.

    ccxt 에러(HTTP 상태나 거래소 코드를 내장하는 경우가 많음), 텔레그램 에러,
    그리고 ``code``/``errno``를 노출하는 일반 예외를 처리한다.
    """
    for attr in ("code", "errno", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if value is not None:
            return str(value)
    return "N/A"


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    context: str,
    **details: Any,
) -> None:
    """예외에 대한 표준화된 실패 레코드를 남긴다.

    콘솔/파일 로그에 실패 컨텍스트, 예외 타입, 추출된 에러 코드, 에러 메시지를
    담은 단일 ERROR 라인을 남기며, 전체 트레이스백도 함께 기록한다
    (``exc_info``를 통해 파일에 기록).

    매개변수
    --------
    logger:
        로그를 출력할 모듈 로거.
    exc:
        포착된 예외 인스턴스.
    context:
        *무엇이* 실패했는지를 나타내는 짧은 기계 판독용 태그.
        예: ``"entry_order"``, ``"network"``, ``"api_call"``, ``"telegram_send"``.
    **details:
        추적성을 위해 로그 라인에 덧붙일 선택적 구조화 키/값 쌍(심볼, 주문 ID 등).
    """
    error_code = _extract_error_code(exc)
    extra = " ".join(f"{key}={value}" for key, value in details.items())
    logger.error(
        "FAILURE | context=%s | type=%s | code=%s | message=%s%s",
        context,
        type(exc).__name__,
        error_code,
        str(exc) or repr(exc),
        f" | {extra}" if extra else "",
        exc_info=True,
    )
