"""FinBERT 주기적(월 1회) 파인튜닝(재학습) 모듈.

운영 중 수집된 뉴스 헤드라인과 감성 라벨을 누적 저장(:func:`record_sample`)하고,
설정된 주기마다(:func:`due_for_run`) 베이스/직전 모델을 이어서 미세 조정한 뒤
``settings.finetune_dir`` 에 저장한다(:func:`run_finetune`). 저장된 모델은
:class:`news_analyzer.SentimentAnalyzer` 가 추론 시 우선 로드한다.

설계 노트
---------
* CPU 환경을 고려해 transformers ``Trainer`` 대신 가벼운 수동 학습 루프를 쓴다.
* 라벨은 운영 중 모델이 부여한 감성 라벨(positive/negative/neutral)을 사용한다.
  실제 체결 손익 등 외부 정답이 있으면 :func:`record_sample` 의 ``label`` 에
  그대로 전달해 교체할 수 있다(아래 호출부만 바꾸면 됨).
* 학습/저장/재로딩 실패는 모두 흡수되어 트레이딩 루프를 막지 않는다.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from config import BASE_DIR, settings
from finbert_paths import finetune_model_dir, resolve_finbert_source
from logger import get_logger, log_exception

log = get_logger(__name__)

# 누적 학습 샘플과 마지막 실행 시각을 보관하는 위치(파인튜닝 디렉터리 옆).
_DATA_DIR = BASE_DIR / "models"
_DATA_PATH = _DATA_DIR / "finetune_samples.jsonl"
_STATE_PATH = _DATA_DIR / "finetune_state.json"

_VALID_LABELS = {"positive", "negative", "neutral"}

# 파일 동시 접근 보호(봇 스레드에서 기록/학습이 함께 일어날 수 있음).
_io_lock = threading.Lock()
_sample_count_cache: tuple[int, float] | None = None  # (count, file mtime)


def _invalidate_sample_count() -> None:
    global _sample_count_cache
    _sample_count_cache = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_sample(text: str, label: str) -> None:
    """학습용 샘플 한 건을 JSONL 에 추가한다(실패는 흡수)."""
    text = (text or "").strip()
    label = (label or "").strip().lower()
    if not text or label not in _VALID_LABELS:
        return
    try:
        with _io_lock:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            with _DATA_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
            global _sample_count_cache
            if _sample_count_cache is not None:
                _sample_count_cache = (_sample_count_cache[0] + 1, _DATA_PATH.stat().st_mtime)
            else:
                _invalidate_sample_count()
    except Exception as exc:  # noqa: BLE001
        log.debug("record_sample skipped | %s: %s", type(exc).__name__, exc)


def sample_count() -> int:
    """현재까지 누적된 학습 샘플 수."""
    global _sample_count_cache
    if not _DATA_PATH.exists():
        return 0
    try:
        mtime = _DATA_PATH.stat().st_mtime
        if _sample_count_cache is not None and _sample_count_cache[1] == mtime:
            return _sample_count_cache[0]
        with _DATA_PATH.open("r", encoding="utf-8") as fh:
            count = sum(1 for _ in fh)
        _sample_count_cache = (count, mtime)
        return count
    except Exception:  # noqa: BLE001
        return 0


def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_state(**kwargs) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = _load_state()
        state.update(kwargs)
        _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.debug("save finetune state skipped | %s: %s", type(exc).__name__, exc)


def due_for_run() -> bool:
    """설정된 재학습 주기(일)가 경과했는지 여부."""
    if not settings.finetune_enabled:
        return False
    state = _load_state()
    last = state.get("last_run")
    if not last:
        # 한 번도 안 했으면 최소 샘플이 모였을 때 첫 실행.
        return sample_count() >= settings.finetune_min_samples
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True
    elapsed_days = (_now() - last_dt).total_seconds() / 86400
    return elapsed_days >= settings.finetune_interval_days


def _read_samples(max_samples: int) -> list[dict]:
    if not _DATA_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        with _io_lock, _DATA_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("text") and obj.get("label") in _VALID_LABELS:
                    rows.append(obj)
    except Exception as exc:  # noqa: BLE001
        log_exception(log, exc, context="finetune_read")
        return []
    return rows[-max_samples:]


def run_finetune() -> bool:
    """누적 샘플로 FinBERT 를 미세 조정하고 ``finetune_dir`` 에 저장한다.

    성공 시 ``True``. 샘플 부족/오류 시 ``False`` 를 반환하며 예외는 흡수한다.
    """
    samples = _read_samples(settings.finetune_max_samples)
    if len(samples) < settings.finetune_min_samples:
        log.info(
            "Fine-tune skipped | samples=%d < min=%d",
            len(samples), settings.finetune_min_samples,
        )
        _save_state(last_skip=_now().isoformat(), last_sample_count=len(samples))
        return False

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(log, exc, context="finetune_import")
        return False

    try:
        source = resolve_finbert_source()
        log.info("Fine-tune start | base=%s | samples=%d", source, len(samples))

        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
        model.train()

        # 모델의 라벨 매핑(소문자)으로 정답 id 구성.
        label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
        texts, labels = [], []
        for s in samples:
            lid = label2id.get(s["label"])
            if lid is None:
                continue
            texts.append(s["text"])
            labels.append(lid)
        if len(texts) < settings.finetune_min_samples:
            log.info("Fine-tune skipped | usable samples=%d", len(texts))
            return False

        device = torch.device("cpu")
        model.to(device)

        # 가벼운 CPU 학습 설정.
        threads = settings.torch_num_threads
        if threads and threads > 0:
            torch.set_num_threads(threads)

        indices = list(range(len(texts)))
        loader = DataLoader(indices, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        epochs = max(1, int(settings.finetune_epochs))
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx in loader:
                idx = [int(i) for i in batch_idx]
                batch_texts = [texts[i] for i in idx]
                batch_labels = torch.tensor([labels[i] for i in idx], dtype=torch.long, device=device)
                enc = tokenizer(
                    batch_texts, return_tensors="pt", truncation=True,
                    max_length=256, padding=True,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                optimizer.zero_grad()
                logits = model(**enc).logits
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            log.info("Fine-tune epoch %d/%d | avg_loss=%.4f", epoch + 1, epochs, total_loss / max(1, len(loader)))

        out_dir = finetune_model_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        _save_state(
            last_run=_now().isoformat(),
            last_trained_samples=len(texts),
            output_dir=str(out_dir),
        )
        log.info("Fine-tune complete | saved=%s | samples=%d", out_dir, len(texts))
        return True
    except Exception as exc:  # noqa: BLE001
        log_exception(log, exc, context="finetune_run")
        return False
