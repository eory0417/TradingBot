"""영문 뉴스 헤드라인 → 한국어 번역 (캐시 + 실패 시 원문 반환)."""

from __future__ import annotations

from functools import lru_cache

from logger import get_logger

log = get_logger(__name__)

_MAX_CHARS = 500
_translators: dict[tuple[str, str], object] = {}


def _get_translator(source: str, target: str):
    """GoogleTranslator 인스턴스를 (source, target) 별로 재사용한다."""
    key = (source, target)
    if key not in _translators:
        from deep_translator import GoogleTranslator

        _translators[key] = GoogleTranslator(source=source, target=target)
    return _translators[key]


@lru_cache(maxsize=512)
def translate_to_korean(text: str) -> str:
    """텍스트를 한국어로 번역한다. 네트워크/라이브러리 오류 시 원문을 반환."""
    return _translate(text, source="auto", target="ko")


@lru_cache(maxsize=512)
def translate_to_english(text: str) -> str:
    """한국어 텍스트를 영어로 번역한다(coinnesskr → FinBERT 입력용).

    네트워크/라이브러리 오류 시 원문을 그대로 반환한다(품질은 저하되지만
    파이프라인이 멈추지 않게 한다).
    """
    return _translate(text, source="ko", target="en")


def _translate(text: str, *, source: str, target: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    snippet = cleaned[:_MAX_CHARS]
    try:
        result = _get_translator(source, target).translate(snippet)
        return (result or snippet).strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("Translation skipped | %s: %s", type(exc).__name__, exc)
        return snippet
