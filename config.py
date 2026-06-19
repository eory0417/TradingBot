"""애플리케이션 중앙 설정 모듈.

``pydantic-settings``를 사용해 모든 런타임 설정을 환경 변수(및 로컬 ``.env``
파일)에서 로드하고 검증한다. 이는 시크릿을 관리하는 12-factor / 프로덕션 표준
방식으로, 설정을 시작 시점에 한 번 검증하고 값이 없거나 잘못된 경우 즉시
실패(fail-fast)하며, 다른 모든 모듈에서 임포트하는 단일 불변 타입 객체
``settings``로 노출한다.

사용 예
-------
    from config import settings

    settings.binance_api_key      # -> SecretStr
    settings.telegram_chat_id     # -> str
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트(이 파일이 위치한 디렉터리). 현재 작업 디렉터리와 무관하게
# .env 파일과 logs 디렉터리 경로를 해석하는 데 사용한다.
BASE_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """강타입(strongly-typed) 애플리케이션 설정.

    시크릿은 ``SecretStr``로 감싸 로그나 트레이스백에 실수로 출력되지 않게 한다
    (``repr``은 ``**********``으로 표시). 실제 사용 지점에서만
    ``.get_secret_value()``를 호출한다.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 바이낸스 API (USDⓈ-M 선물) ----
    binance_api_key: SecretStr = Field(..., alias="BINANCE_API_KEY")
    binance_secret_key: SecretStr = Field(..., alias="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(default=True, alias="BINANCE_TESTNET")

    # ---- 텔레그램 ----
    telegram_token: SecretStr = Field(..., alias="TELEGRAM_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")

    # ---- 로깅 ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    # ---- 뉴스 수집 & NLP ----
    # CryptoPanic API 토큰은 선택 사항이며, 비어 있으면 분석기는 무료 공개 RSS
    # 피드로 자동 전환한다.
    cryptopanic_api_token: SecretStr = Field(
        default=SecretStr(""), alias="CRYPTOPANIC_API_TOKEN"
    )
    # 뉴스 수집 폴링 주기(초 단위, 기본값 = 1분).
    news_poll_interval: int = Field(default=60, alias="NEWS_POLL_INTERVAL")
    # 기관급 금융 감성 분석 모델의 Hugging Face 모델 ID.
    finbert_model: str = Field(default="ProsusAI/finbert", alias="FINBERT_MODEL")
    # 추론 시 torch가 사용할 CPU 스레드 수(0 = 자동/전체 코어).
    torch_num_threads: int = Field(default=0, alias="TORCH_NUM_THREADS")

    # ---- 트레이딩 엔진 (3단계) ----
    # 매매 대상 코인 심볼(USDⓈ-M 선물, 쉼표 구분 환경변수로 재정의 가능).
    trade_symbols: str = Field(default="BTC,ETH,SOL,XRP", alias="TRADE_SYMBOLS")
    # 지표 계산용 캔들 타임프레임.
    timeframe: str = Field(default="15m", alias="TIMEFRAME")
    # 동시 보유 가능한 최대 포지션 수.
    max_positions: int = Field(default=2, alias="MAX_POSITIONS")
    # 가변형 시장 지정가 주문의 체결 조건(IOC 또는 FOK).
    order_time_in_force: str = Field(default="IOC", alias="ORDER_TIME_IN_FORCE")
    # 1회 진입 명목 가치(USDT 기준).
    position_size_usdt: float = Field(default=50.0, alias="POSITION_SIZE_USDT")
    # 주문 레버리지 배수.
    leverage: int = Field(default=3, alias="LEVERAGE")
    # 증거금 모드(격리: isolated / 교차: cross).
    margin_mode: str = Field(default="isolated", alias="MARGIN_MODE")

    # ---- 익절/손절 전략 (4단계) ----
    # 고정 손절 비율(%). 진입가 대비 이 비율만큼 불리하게 움직이면 시장가 청산.
    stop_loss_pct: float = Field(default=2.0, alias="STOP_LOSS_PCT")
    # 동적 익절(Trailing Stop) 기본 ATR 배수.
    trailing_atr_mult: float = Field(default=3.0, alias="TRAILING_ATR_MULT")
    # 강한 추세/뉴스 신호 발생 시 적용하는 축소된 ATR 배수(익절 라인을 바짝 당김).
    trailing_atr_mult_tight: float = Field(default=1.5, alias="TRAILING_ATR_MULT_TIGHT")
    # 실시간 뉴스 가중치 축소 트리거 임계값(긍정 0.7 / 부정 -0.7).
    news_score_threshold: float = Field(default=0.7, alias="NEWS_SCORE_THRESHOLD")
    # 횡보 시 시간 청산까지의 보유 시간(시간 단위).
    time_exit_hours: float = Field(default=7.0, alias="TIME_EXIT_HOURS")
    # 포지션 모니터링 주기(초).
    monitor_interval: int = Field(default=15, alias="MONITOR_INTERVAL")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.strip().upper()
        if normalized not in valid:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(valid)}, got {value!r}"
            )
        return normalized

    @field_validator("telegram_chat_id")
    @classmethod
    def _validate_chat_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("TELEGRAM_CHAT_ID must not be empty")
        return value.strip()

    @field_validator("news_poll_interval")
    @classmethod
    def _validate_poll_interval(cls, value: int) -> int:
        if value < 10:
            raise ValueError("NEWS_POLL_INTERVAL must be >= 10 seconds")
        return value

    @field_validator("order_time_in_force")
    @classmethod
    def _validate_tif(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"IOC", "FOK"}:
            raise ValueError("ORDER_TIME_IN_FORCE must be 'IOC' or 'FOK'")
        return normalized

    @field_validator("margin_mode")
    @classmethod
    def _validate_margin_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"isolated", "cross"}:
            raise ValueError("MARGIN_MODE must be 'isolated' or 'cross'")
        return normalized

    @property
    def cryptopanic_token(self) -> str:
        """CryptoPanic 토큰을 평문 문자열로 반환(미설정 시 '')."""
        return self.cryptopanic_api_token.get_secret_value().strip()

    @property
    def symbols(self) -> list[str]:
        """매매 대상 심볼을 ccxt 형식('BTC/USDT')의 리스트로 반환한다."""
        result: list[str] = []
        for raw in self.trade_symbols.split(","):
            token = raw.strip().upper()
            if not token:
                continue
            # 'BTC' -> 'BTC/USDT', 이미 페어 형태면 그대로 사용.
            result.append(token if "/" in token else f"{token}/USDT")
        return result

    @property
    def log_path(self) -> Path:
        """로그 디렉터리의 절대 경로(로거가 지연 생성)."""
        path = Path(self.log_dir)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """검증된 싱글톤 설정 인스턴스를 반환한다.

    캐싱되므로 프로세스당 ``.env`` 파일은 정확히 한 번만 파싱·검증된다. 필수
    시크릿이 누락되거나 형식이 잘못되면 즉시 ``pydantic.ValidationError``를
    발생시킨다.
    """
    return Settings()  # type: ignore[call-arg]


# ``from config import settings`` 편의를 위해 즉시 인스턴스화한 싱글톤.
settings: Settings = get_settings()
