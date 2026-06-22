"""뉴스 수집 + 기관급 금융 감성 분석 모듈 (2단계).

서로 협력하는 세 가지 구성요소를 제공한다.

  * :class:`NewsCollector`   - CryptoPanic API(토큰이 설정된 경우) 또는 무료 공개
    RSS 피드(기본 16+개)를 비동기로 폴링하며, URL/제목 기준 중복 제거 후
    이미 처리한 항목은 seen 으로 필터한다.
  * :class:`SentimentAnalyzer` - 기관급 금융 특화 감성 모델(기본
    ``ProsusAI/finbert``)을 로드하여 영문 텍스트를 ``-1.0``(매우 부정)에서
    ``+1.0``(매우 긍정)까지 연속 점수로 평가한다. 추론은 일반 CPU에 맞춰
    최적화한다(스레드 튜닝 + INT8 동적 양자화 + ``inference_mode``).
  * :class:`NewsAnalyzer`    - 1분 주기 폴링 루프를 오케스트레이션하여 새 뉴스를
    수집·점수화하고, 결과 ``AnalyzedNews``를 콜백으로 전달한다.
    **시작 직후 첫 폴링은 워밍업**으로 피드 기사를 ``seen`` 에 등록하되, 최근
    기사는 GUI에 표시한다(진입은 :mod:`bot` 의 grace·발행 시각 규칙으로 제한).

사용 예
-------
    import asyncio
    from news_analyzer import NewsAnalyzer

    async def on_news(item):
        print(item.score, item.title)

    async def main():
        analyzer = NewsAnalyzer()
        await analyzer.start(on_news)   # 영구 실행, 매분 폴링

    asyncio.run(main())

또는 단일 문장을 직접 점수화:

    from news_analyzer import SentimentAnalyzer
    sa = SentimentAnalyzer()
    sa.load()
    print(sa.score("Bitcoin ETF approved, market rallies"))  # ~ +0.9
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable
from urllib.parse import urlparse, urlunparse

import aiohttp
import feedparser

from config import settings
from logger import get_logger, log_exception
from translator import translate_to_english

log = get_logger(__name__)

# CryptoPanic 토큰이 설정되지 않았을 때 사용하는 무료 공개 암호화폐 RSS 피드.
DEFAULT_RSS_FEEDS: tuple[str, ...] = (
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/feed",
    "https://www.theblock.co/rss.xml",
    "https://blockworks.co/feed/",
    "https://u.today/rss",
    "https://news.bitcoin.com/feed/",
    "https://beincrypto.com/feed/",
    "https://www.newsbtc.com/feed/",
    "https://ambcrypto.com/feed/",
    "https://cryptopotato.com/feed/",
    "https://coinjournal.net/feed/",
    "https://crypto.news/feed/",
    "https://bitcoinist.com/feed/",
    "https://cryptoslate.com/feed/",
    "https://dailyhodl.com/feed/",
    "https://www.cryptoglobe.com/latest/feed/",
)

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# 정중한 수집을 위한 HTTP 타임아웃/헤더.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=20)
_HTTP_HEADERS = {"User-Agent": "NewsTradingBot/1.0 (+https://example.local)"}


def _make_http_session() -> aiohttp.ClientSession:
    """RSS/aiohttp DNS — exe 환경에서 aiodns 실패 시 시스템 DNS 사용."""
    from aiohttp.resolver import ThreadedResolver

    connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
    return aiohttp.ClientSession(headers=_HTTP_HEADERS, connector=connector, trust_env=True)


# --------------------------------------------------------------------------- #
#  데이터 모델
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class NewsItem:
    """정규화된 단일 뉴스 헤드라인."""

    id: str
    title: str
    url: str
    source: str
    published_at: datetime
    origin: str = "unknown"  # rss | cryptopanic | coinnesskr
    # coinnesskr 원문(한국어) 헤드라인. FinBERT 입력은 EN으로 번역해 title 에 채운다.
    title_ko: str = ""


@dataclass(slots=True)
class AnalyzedNews:
    """[-1.0, 1.0] 감성 점수가 부여된 뉴스 항목."""

    item: NewsItem
    score: float
    label: str
    probabilities: dict[str, float] = field(default_factory=dict)

    # 편의 위임 프로퍼티.
    @property
    def title(self) -> str:
        return self.item.title

    @property
    def url(self) -> str:
        return self.item.url


# --------------------------------------------------------------------------- #
#  뉴스 수집(비동기)
# --------------------------------------------------------------------------- #
class NewsCollector:
    """암호화폐 뉴스 헤드라인을 비동기로 수집·중복 제거한다."""

    def __init__(
        self,
        rss_feeds: Iterable[str] | None = None,
        max_seen: int = 5000,
    ) -> None:
        self._token = settings.cryptopanic_token
        override = settings.rss_feed_urls
        if rss_feeds is not None:
            self._rss_feeds = tuple(rss_feeds)
        elif override:
            self._rss_feeds = override
        else:
            self._rss_feeds = DEFAULT_RSS_FEEDS

        # 메모리 무한 증가를 막기 위한 한도 있는 LRU 형태의 seen 집합.
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._max_seen = max_seen
        self._source_mode = "cryptopanic" if settings.use_cryptopanic else "rss"
        if settings.use_cryptopanic and not self._token:
            log.warning(
                "NEWS_SOURCE_MODE=cryptopanic 이지만 CRYPTOPANIC_API_TOKEN 이 없습니다 — "
                "뉴스가 수집되지 않습니다."
            )
        log.info(
            "NewsCollector ready | mode=%s | feeds=%d",
            self._source_mode,
            len(self._rss_feeds),
        )

    # ---- 공개 API ----
    async def fetch_new(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        """이전에 본 적 없는 뉴스 항목만 반환한다(중복 제거)."""
        items = await self.fetch_all(session)
        fresh = [it for it in items if not self._is_seen(it.id)]
        for it in fresh:
            self._mark_seen(it.id)
        if fresh:
            by_origin: dict[str, int] = {}
            for it in fresh:
                by_origin[it.origin] = by_origin.get(it.origin, 0) + 1
            parts = " ".join(f"{k}={v}" for k, v in sorted(by_origin.items()))
            log.info(
                "Collected %d new headline(s) | mode=%s | %s",
                len(fresh),
                self._source_mode,
                parts,
            )
        return fresh

    async def fetch_all(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        """모든 소스에서 헤드라인을 수집·병합한다(seen 갱신 없음)."""
        if settings.use_cryptopanic:
            return await self._fetch_cryptopanic(session)
        items = await self._fetch_rss(session)
        return _dedupe_items(items)

    def seed_seen(self, items: list[NewsItem]) -> int:
        """워밍업: 목록의 모든 ID 를 seen 에 등록한다."""
        for it in items:
            self._mark_seen(it.id)
        return len(items)

    # ---- 중복 제거 ----
    def _is_seen(self, item_id: str) -> bool:
        return item_id in self._seen

    def _mark_seen(self, item_id: str) -> None:
        self._seen[item_id] = None
        if len(self._seen) > self._max_seen:
            self._seen.popitem(last=False)  # 가장 오래된 항목 제거

    # ---- CryptoPanic 소스 ----
    async def _fetch_cryptopanic(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        params = {"auth_token": self._token, "public": "true", "kind": "news"}
        try:
            async with session.get(
                CRYPTOPANIC_URL, params=params, timeout=_HTTP_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except aiohttp.ClientError as exc:
            log_exception(log, exc, context="news_fetch", source="cryptopanic")
            return []
        except Exception as exc:  # noqa: BLE001
            log_exception(log, exc, context="news_fetch", source="cryptopanic")
            return []

        items: list[NewsItem] = []
        for post in payload.get("results", []):
            title = (post.get("title") or "").strip()
            if not title:
                continue
            url = post.get("url", "") or ""
            raw_id = str(post.get("id") or _hash(title))
            items.append(
                NewsItem(
                    id=_stable_item_id(url, title, raw_id),
                    title=title,
                    url=url,
                    source=(post.get("source") or {}).get("title", "CryptoPanic"),
                    published_at=_parse_dt(post.get("published_at")),
                    origin="cryptopanic",
                )
            )
        return items

    # ---- RSS 소스 ----
    async def _fetch_rss(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        tasks = [self._fetch_one_feed(session, url) for url in self._rss_feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items: list[NewsItem] = []
        for result in results:
            if isinstance(result, BaseException):
                log_exception(log, result, context="news_fetch", source="rss")
                continue
            items.extend(result)
        return items

    async def _fetch_one_feed(
        self, session: aiohttp.ClientSession, url: str
    ) -> list[NewsItem]:
        async with session.get(url, timeout=_HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            raw = await resp.read()
        # feedparser는 블로킹/CPU 바운드이므로 이벤트 루프 밖에서 실행한다.
        parsed = await asyncio.to_thread(feedparser.parse, raw)
        source = parsed.feed.get("title", url) if parsed.feed else url

        items: list[NewsItem] = []
        for entry in parsed.entries:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            link = entry.get("link", "") or ""
            uid = entry.get("id") or link or _hash(title)
            items.append(
                NewsItem(
                    id=_stable_item_id(link, title, str(uid)),
                    title=title,
                    url=link,
                    source=source,
                    published_at=_entry_dt(entry),
                    origin="rss",
                )
            )
        return items


# --------------------------------------------------------------------------- #
#  감성 분석(FinBERT, CPU 최적화)
# --------------------------------------------------------------------------- #
class SentimentAnalyzer:
    """[-1, 1] 점수를 반환하는 기관급 금융 특화 감성 모델.

    로딩은 지연 처리된다(:meth:`load` 호출 또는 최초 :meth:`score` 시 자동 로드).
    추론은 가장 가볍고 빠른 CPU 사용량을 위해 INT8 동적 양자화, 제한된 CPU
    스레드, ``torch.inference_mode``로 수행한다.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.finbert_model
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._id2label: dict[int, str] = {}
        self._loaded = False

    def _resolve_source(self) -> str:
        """추론에 사용할 모델 경로: 파인튜닝 모델이 있으면 우선, 없으면 원본."""
        try:
            finetuned = settings.finetune_path
            if (finetuned / "config.json").exists():
                return str(finetuned)
        except Exception:  # noqa: BLE001
            pass
        return self._model_name

    def reload(self) -> None:
        """현재 모델을 내리고 (파인튜닝본 우선) 다시 로드한다."""
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self.load()

    def load(self) -> None:
        """모델을 다운로드(캐시됨)하고 CPU 추론용으로 준비한다."""
        if self._loaded:
            return

        # 앱 나머지 부분이 torch 로딩 없이 시작될 수 있도록 지연 임포트한다.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # ---- CPU 스레드 튜닝 ----
        threads = settings.torch_num_threads
        if threads and threads > 0:
            torch.set_num_threads(threads)

        source = self._resolve_source()
        if source != self._model_name:
            log.info("Using fine-tuned sentiment model | path=%s", source)
        log.info(
            "Loading sentiment model '%s' | torch_threads=%d",
            source,
            torch.get_num_threads(),
        )

        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
        model.eval()

        # ---- INT8 동적 양자화: CPU에서 가장 가볍고 빠름 ----
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            log.info("Applied INT8 dynamic quantization to Linear layers")
        except Exception as exc:  # noqa: BLE001 - 양자화는 최선 노력(best-effort)
            log_exception(log, exc, context="model_quantize")

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._id2label = {
            int(k): str(v).lower() for k, v in model.config.id2label.items()
        }
        self._loaded = True
        log.info("Sentiment model ready | labels=%s", list(self._id2label.values()))

    def score(self, text: str) -> float:
        """단일 문장에 대한 [-1.0, 1.0] 감성 점수를 반환한다."""
        return self.analyze(text)[0]

    def analyze(self, text: str) -> tuple[float, str, dict[str, float]]:
        """``text``에 대해 ``(점수, 라벨, 확률분포)``를 반환한다.

        ``점수 = P(positive) - P(negative)`` ∈ [-1, 1]이며, ``라벨``은 가장 높은
        확률을 가진 클래스다.
        """
        if not self._loaded:
            self.load()
        assert self._torch is not None and self._model is not None

        text = (text or "").strip()
        if not text:
            return 0.0, "neutral", {}

        torch = self._torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        prob_map = {
            self._id2label[i]: float(probs[i]) for i in range(len(probs))
        }
        pos = prob_map.get("positive", 0.0)
        neg = prob_map.get("negative", 0.0)
        score = round(pos - neg, 4)
        label = max(prob_map, key=prob_map.get)
        return score, label, {k: round(v, 4) for k, v in prob_map.items()}


# --------------------------------------------------------------------------- #
#  오케스트레이션: 매분 폴링, 수집 + 분석
# --------------------------------------------------------------------------- #
NewsCallback = Callable[[AnalyzedNews], Awaitable[None]]


class NewsAnalyzer:
    """``NEWS_POLL_INTERVAL``초마다 뉴스를 폴링하고 헤드라인을 점수화한다."""

    def __init__(self) -> None:
        self.collector = NewsCollector()
        self.sentiment = SentimentAnalyzer()
        self._interval = settings.news_poll_interval
        self._running = False
        # 시작 직후 첫 폴링: 피드에 남은 기존 기사는 seen 만 등록, 진입/콜백 생략.
        self._warmup_pending = True
        self._last_warmup_count = 0
        self._on_status: Callable[[str], None] | None = None
        # 소스 간 중복 진입 방지(예: RSS·coinnesskr 가 같은 coinness.com URL 전달).
        self._dispatched: "OrderedDict[str, None]" = OrderedDict()
        self._dispatch_max = 5000

    async def start(
        self,
        callback: NewsCallback,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        """활성화된 소스 태스크를 실행하며, 분석된 항목마다 ``callback``을 호출한다."""

        def _status(msg: str) -> None:
            log.info(msg)
            if on_status is not None:
                on_status(msg)

        self._on_status = on_status
        # 첫 폴링이 빠르도록 루프 시작 전에 모델을 한 번 워밍업한다.
        _status("FinBERT 모델 로딩 중 (첫 실행 시 ~438MB 다운로드, 완료 후 뉴스 수집 시작)")
        await asyncio.to_thread(self.sentiment.load)
        _status(f"FinBERT 로딩 완료 — 뉴스 소스: {settings.news_source_mode}")
        self._running = True

        tasks: list[asyncio.Task] = []
        if settings.use_rss or settings.use_cryptopanic:
            tasks.append(asyncio.create_task(self._poll_loop(callback)))
        if settings.use_coinnesskr:
            tasks.append(asyncio.create_task(self._coinness_loop(callback)))

        if not tasks:
            log.warning("활성화된 뉴스 소스가 없습니다 | mode=%s", settings.news_source_mode)
            return

        log.info(
            "NewsAnalyzer started | mode=%s | tasks=%d | interval=%ds",
            settings.news_source_mode,
            len(tasks),
            self._interval,
        )
        await asyncio.gather(*tasks)

    async def _poll_loop(self, callback: NewsCallback) -> None:
        """RSS/CryptoPanic 주기 폴링 루프."""
        async with _make_http_session() as session:
            while self._running:
                started = asyncio.get_event_loop().time()
                try:
                    await self._poll_once(session, callback)
                except Exception as exc:  # noqa: BLE001 - 루프는 살아남아야 한다
                    log_exception(log, exc, context="news_loop")

                # 작업 소요 시간을 차감하고 남은 주기만큼 대기한다.
                elapsed = asyncio.get_event_loop().time() - started
                await asyncio.sleep(max(1.0, self._interval - elapsed))

    async def _coinness_loop(self, callback: NewsCallback) -> None:
        """coinnesskr(Telethon) 실시간 수신 루프."""
        from telegram_news import CoinnessListener

        listener = CoinnessListener()
        ok = await listener.connect()
        if not ok:
            if self._on_status is not None:
                self._on_status("coinnesskr 비활성 — 세션/자격증명을 확인하세요 (telegram_login.py)")
            return

        # 워밍업: 최근 메시지를 seen 등록하고 일부는 GUI 표시(진입은 bot 에서 제한).
        warm = await listener.warmup_recent(settings.news_warmup_display_limit)
        if warm:
            for it in warm:
                self._mark_dispatched(it)
            display = _items_for_warmup_display(warm)
            log.info("coinnesskr warmup | recent=%d display=%d", len(warm), len(display))
            if display:
                await self._analyze_and_dispatch(display, callback, dedup=False)

        listener.add_handler(lambda item: self._analyze_and_dispatch([item], callback))
        if self._on_status is not None:
            self._on_status(f"coinnesskr 수신 시작 (@{settings.coinness_channel})")

        try:
            while self._running:
                await asyncio.sleep(1.0)
        finally:
            await listener.stop()

    def stop(self) -> None:
        self._running = False

    def _mark_dispatched(self, item: NewsItem) -> bool:
        """이미 처리한 항목이면 ``True``. 아니면 등록 후 ``False`` 를 반환한다."""
        key = _stable_item_id(item.url, item.title_ko or item.title, item.id)
        if key in self._dispatched:
            return True
        self._dispatched[key] = None
        if len(self._dispatched) > self._dispatch_max:
            self._dispatched.popitem(last=False)
        return False

    async def poll_once(self, callback: NewsCallback) -> list[AnalyzedNews]:
        """수집+분석 사이클을 1회 실행한다(테스트/수동 실행에 유용)."""
        if not self.sentiment._loaded:
            await asyncio.to_thread(self.sentiment.load)
        async with _make_http_session() as session:
            return await self._poll_once(session, callback)

    async def _analyze_and_dispatch(
        self,
        items: list[NewsItem],
        callback: NewsCallback,
        *,
        dedup: bool = True,
    ) -> list[AnalyzedNews]:
        analyzed: list[AnalyzedNews] = []
        for item in items:
            if dedup and self._mark_dispatched(item):
                continue
            # coinnesskr 는 한국어 → 영어로 번역해 FinBERT/심볼 탐지 입력을 통일한다.
            if item.origin == "coinnesskr":
                en = await asyncio.to_thread(
                    translate_to_english, item.title_ko or item.title
                )
                if en:
                    item.title = en
            score, label, probs = await asyncio.to_thread(
                self.sentiment.analyze, item.title
            )
            result = AnalyzedNews(item=item, score=score, label=label, probabilities=probs)
            analyzed.append(result)
            try:
                await callback(result)
            except Exception as exc:  # noqa: BLE001 - 콜백이 루프를 죽이면 안 됨
                log_exception(log, exc, context="news_callback", title=item.title)
        return analyzed

    async def _poll_once(
        self, session: aiohttp.ClientSession, callback: NewsCallback
    ) -> list[AnalyzedNews]:
        # ---- 시작 워밍업: seen 등록 + 최근 기사는 GUI 표시(진입은 bot 에서 제한) ----
        if self._warmup_pending:
            all_items = await self.collector.fetch_all(session)
            if not all_items:
                log.warning("News warmup: feed empty — next poll will retry seeding")
                return []
            self.collector.seed_seen(all_items)
            self._warmup_pending = False
            self._last_warmup_count = len(all_items)
            display_items = _items_for_warmup_display(all_items)
            log.info(
                "News warmup complete | seeded=%d display=%d",
                len(all_items),
                len(display_items),
            )
            if self._on_status is not None:
                self._on_status(
                    f"뉴스 워밍업 완료 ({len(all_items)}건 등록, "
                    f"최근 {len(display_items)}건 화면 표시)"
                )
            if display_items:
                return await self._analyze_and_dispatch(display_items, callback)
            return []

        items = await self.collector.fetch_new(session)
        if items:
            log.info("Processing %d new headline(s) for display", len(items))
        return await self._analyze_and_dispatch(items, callback)


# --------------------------------------------------------------------------- #
#  헬퍼
# --------------------------------------------------------------------------- #
def _items_for_warmup_display(items: list[NewsItem]) -> list[NewsItem]:
    """워밍업 직후 GUI에 보여줄 최근 기사 목록(진입 여부와 무관)."""
    limit = settings.news_warmup_display_limit
    max_age = settings.news_max_age_minutes
    now = datetime.now(timezone.utc)
    sorted_items = sorted(items, key=lambda it: it.published_at, reverse=True)

    if max_age > 0:
        recent: list[NewsItem] = []
        for it in sorted_items:
            pub = it.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if (now - pub).total_seconds() <= max_age * 60:
                recent.append(it)
        if recent:
            return recent[:limit]

    return sorted_items[:limit]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url.lower())
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _stable_item_id(url: str, title: str, fallback: str) -> str:
    """소스 간 중복 제거·seen 키로 쓸 안정 ID(URL 우선, 없으면 제목 해시)."""
    norm_url = _normalize_url(url)
    if norm_url:
        return norm_url
    norm_title = _normalize_title(title)
    if norm_title:
        return f"title:{_hash(norm_title)}"
    return fallback


def _dedupe_items(items: list[NewsItem]) -> list[NewsItem]:
    """URL·제목 기준 중복 제거. 동일 기사면 published_at 이 더 이른 항목 유지."""
    by_key: dict[str, NewsItem] = {}
    for item in items:
        norm_url = _normalize_url(item.url)
        key = norm_url or f"title:{_hash(_normalize_title(item.title))}"
        existing = by_key.get(key)
        if existing is None or item.published_at < existing.published_at:
            by_key[key] = item
    return list(by_key.values())


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _entry_dt(entry) -> datetime:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    # 날짜 없음 → 아주 오래된 것으로 간주(신선도 필터에서 진입 제외).
    return datetime(1970, 1, 1, tzinfo=timezone.utc)
