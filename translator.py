"""영문 뉴스 헤드라인 → 한국어 번역 (캐시 + 실패 시 원문 반환)."""

from __future__ import annotations

from functools import lru_cache

from logger import get_logger

log = get_logger(__name__)

_MAX_CHARS = 500


@lru_cache(maxsize=512)
def translate_to_korean(text: str) -> str:
    """텍스트를 한국어로 번역한다. 네트워크/라이브러리 오류 시 원문을 반환."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    snippet = cleaned[:_MAX_CHARS]
    try:
        from deep_translator import GoogleTranslator

        result = GoogleTranslator(source="auto", target="ko").translate(snippet)
        return (result or snippet).strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("Translation skipped | %s: %s", type(exc).__name__, exc)
        return snippet
