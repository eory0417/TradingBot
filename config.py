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

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트. PyInstaller exe는 _internal 이 아닌 exe 옆 폴더를 기준으로
# .env · logs · models 경로를 해석한다.
if getattr(sys, "frozen", False):
    BASE_DIR: Path = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent


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

    # ---- 텔레그램 (알림 발송용 봇) ----
    telegram_token: SecretStr = Field(..., alias="TELEGRAM_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")

    # ---- 텔레그램 (coinnesskr 수신용 유저 세션, Telethon) ----
    # 알림 발송 봇(telegram_token)과 완전히 별개다. https://my.telegram.org 에서 발급.
    telegram_api_id: int = Field(default=0, alias="TELEGRAM_API_ID")
    telegram_api_hash: SecretStr = Field(
        default=SecretStr(""), alias="TELEGRAM_API_HASH"
    )
    # 세션 파일 이름(확장자 제외). 실제 파일은 models/<name>.session 으로 저장된다.
    telegram_session_name: str = Field(
        default="tradingbot_plus", alias="TELEGRAM_SESSION_NAME"
    )
    # 수신 대상 텔레그램 채널 username(@ 제외).
    # 한국어: coinnesskr · 영어: coinnessGL
    coinness_channel: str = Field(default="coinnessGL", alias="COINNESS_CHANNEL")
    # 채널 언어: ko(한국어 → 영어 번역 후 분석) | en(영어 원문 그대로 분석).
    coinness_lang: str = Field(default="en", alias="COINNESS_LANG")

    # ---- 로깅 ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    # ---- 뉴스 수집 & NLP ----
    # CryptoPanic API 토큰은 선택 사항(설정 시 CryptoPanic 단독 모드 우선).
    cryptopanic_api_token: SecretStr = Field(
        default=SecretStr(""), alias="CRYPTOPANIC_API_TOKEN"
    )
    # 뉴스 소스 모드: rss | coinnesskr | rss_coinnesskr | cryptopanic.
    news_source_mode: str = Field(
        default="rss_coinnesskr", alias="NEWS_SOURCE_MODE"
    )
    # 선택: 쉼표 구분 RSS URL 오버라이드(비어 있으면 기본 16+ 피드 목록).
    news_rss_feeds: str = Field(default="", alias="NEWS_RSS_FEEDS")
    # 뉴스 수집 폴링 주기(초 단위, 기본값 = 1분).
    news_poll_interval: int = Field(default=60, alias="NEWS_POLL_INTERVAL")
    # 시작 후 이 시간(초)이 지나기 전·첫 RSS 워밍업 전에는 뉴스로 진입하지 않는다.
    news_entry_grace_seconds: int = Field(default=60, alias="NEWS_ENTRY_GRACE_SEC")
    # 진입 허용 최대 기사 나이(분). 발행 시각이 이보다 오래되면 진입하지 않는다.
    news_max_age_minutes: float = Field(default=30.0, alias="NEWS_MAX_AGE_MINUTES")
    # 워밍업 직후 GUI에 표시할 최근 RSS 기사 수(진입은 grace·발행시각으로 별도 제한).
    news_warmup_display_limit: int = Field(default=30, alias="NEWS_WARMUP_DISPLAY_LIMIT")
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
    # 주문 레버리지 배수(자동 레버리지 비활성 시 사용하는 수동 값).
    leverage: int = Field(default=3, alias="LEVERAGE")
    # 자동 레버리지: True면 뉴스 점수 강도에 따라 레버리지를 결정한다.
    auto_leverage: bool = Field(default=True, alias="AUTO_LEVERAGE")
    # 증거금 모드(격리: isolated / 교차: cross).
    margin_mode: str = Field(default="isolated", alias="MARGIN_MODE")

    # ---- 익절/손절 전략 (4단계) ----
    # 고정 손절 비율(%). 진입가 대비 이 비율만큼 불리하게 움직이면 시장가 청산.
    stop_loss_pct: float = Field(default=2.0, alias="STOP_LOSS_PCT")
    # 동적 익절(Trailing Stop) 기본 ATR 배수.
    trailing_atr_mult: float = Field(default=3.0, alias="TRAILING_ATR_MULT")
    # 강한 추세/뉴스 신호 발생 시 적용하는 축소된 ATR 배수(익절 라인을 바짝 당김).
    trailing_atr_mult_tight: float = Field(default=1.5, alias="TRAILING_ATR_MULT_TIGHT")
    # Trailing stop 활성화 최소 이익(%). 이 수익률 이상일 때만 트레일링 익절이 동작한다.
    trailing_profit_pct: float = Field(default=2.0, alias="TRAILING_PROFIT_PCT")
    # 실시간 뉴스 가중치 축소 트리거 임계값(긍정 0.7 / 부정 -0.7).
    news_score_threshold: float = Field(default=0.7, alias="NEWS_SCORE_THRESHOLD")
    # 횡보 시 시간 청산까지의 보유 시간(시간 단위).
    time_exit_hours: float = Field(default=7.0, alias="TIME_EXIT_HOURS")
    # 포지션 모니터링 주기(초).
    monitor_interval: int = Field(default=15, alias="MONITOR_INTERVAL")

    # ---- FinBERT 주기적 파인튜닝(재학습) ----
    # 자동 재학습 활성화 여부.
    finetune_enabled: bool = Field(default=True, alias="FINETUNE_ENABLED")
    # 재학습 주기(일). 기본 30일 = 월 1회.
    finetune_interval_days: int = Field(default=30, alias="FINETUNE_INTERVAL_DAYS")
    # 재학습을 시도하기 위한 최소 누적 샘플 수.
    finetune_min_samples: int = Field(default=50, alias="FINETUNE_MIN_SAMPLES")
    # 재학습에 사용할 최근 샘플 최대 개수(메모리/시간 절약).
    finetune_max_samples: int = Field(default=1000, alias="FINETUNE_MAX_SAMPLES")
    # 재학습 epoch 수(CPU 환경이므로 작게 유지).
    finetune_epochs: int = Field(default=1, alias="FINETUNE_EPOCHS")
    # 파인튜닝된 모델을 저장/로드할 디렉터리(프로젝트 루트 기준 상대경로 허용).
    finetune_dir: str = Field(default="models/finbert-finetuned", alias="FINETUNE_DIR")

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

    @field_validator("news_entry_grace_seconds")
    @classmethod
    def _validate_entry_grace(cls, value: int) -> int:
        if value < 0:
            raise ValueError("NEWS_ENTRY_GRACE_SEC must be >= 0")
        return value

    @field_validator("news_max_age_minutes")
    @classmethod
    def _validate_news_max_age(cls, value: float) -> float:
        if value < 0:
            raise ValueError("NEWS_MAX_AGE_MINUTES must be >= 0")
        return value

    @field_validator("news_warmup_display_limit")
    @classmethod
    def _validate_warmup_display(cls, value: int) -> int:
        if value < 1:
            raise ValueError("NEWS_WARMUP_DISPLAY_LIMIT must be >= 1")
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

    @field_validator("news_source_mode")
    @classmethod
    def _validate_news_source_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"rss", "coinnesskr", "rss_coinnesskr", "cryptopanic"}
        if normalized not in valid:
            raise ValueError(
                f"NEWS_SOURCE_MODE must be one of {sorted(valid)}, got {value!r}"
            )
        return normalized

    @field_validator("coinness_lang")
    @classmethod
    def _validate_coinness_lang(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"ko", "en"}:
            raise ValueError("COINNESS_LANG must be 'ko' or 'en'")
        return normalized

    @property
    def cryptopanic_token(self) -> str:
        """CryptoPanic 토큰을 평문 문자열로 반환(미설정 시 '')."""
        return self.cryptopanic_api_token.get_secret_value().strip()

    @property
    def coinness_is_english(self) -> bool:
        """coinness 채널이 영어 원문(번역 불필요)인지 여부."""
        return self.coinness_lang == "en"

    @property
    def use_rss(self) -> bool:
        """현재 모드에서 RSS 폴링을 사용하는지 여부."""
        return self.news_source_mode in {"rss", "rss_coinnesskr"}

    @property
    def use_coinnesskr(self) -> bool:
        """현재 모드에서 coinnesskr(Telethon) 수신을 사용하는지 여부."""
        return self.news_source_mode in {"coinnesskr", "rss_coinnesskr"}

    @property
    def use_cryptopanic(self) -> bool:
        """현재 모드에서 CryptoPanic API를 사용하는지 여부."""
        return self.news_source_mode == "cryptopanic"

    @property
    def telegram_api_hash_value(self) -> str:
        """Telethon API hash 평문 문자열(미설정 시 '')."""
        return self.telegram_api_hash.get_secret_value().strip()

    @property
    def telegram_session_path(self) -> Path:
        """Telethon 세션 파일의 절대 경로(models/<name>.session)."""
        return BASE_DIR / "models" / f"{self.telegram_session_name}.session"

    @property
    def rss_feed_urls(self) -> tuple[str, ...]:
        """RSS 피드 URL 목록(NEWS_RSS_FEEDS 오버라이드 또는 기본값)."""
        raw = self.news_rss_feeds.strip()
        if not raw:
            return ()
        return tuple(u.strip() for u in raw.split(",") if u.strip())

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

    @property
    def finetune_path(self) -> Path:
        """파인튜닝 모델 저장 디렉터리의 절대 경로."""
        path = Path(self.finetune_dir)
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
