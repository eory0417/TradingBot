"""FinBERT 모델 경로 결정 (추론·재학습 공용).

``news_analyzer``(감성 분석)와 ``finetune``(월간 재학습)이 각각 "파인튜닝본이
있으면 그걸 쓰고, 없으면 원본 ProsusAI/finbert" 를 따로 구현하면 규칙이 어긋날
수 있다. 이 모듈이 **한 곳**에서 경로를 정한다.
"""

from __future__ import annotations

from pathlib import Path

from config import settings


def finetune_model_dir() -> Path:
    """파인튜닝 모델이 저장되는 디렉터리 (절대 경로)."""
    return settings.finetune_path


def finetuned_model_exists() -> bool:
    """저장된 파인튜닝 가중치가 있는지 (config.json 존재 여부)."""
    return (finetune_model_dir() / "config.json").exists()


def resolve_finbert_source() -> str:
    """추론·재학습에 쓸 FinBERT 경로 또는 Hugging Face 모델 ID.

    우선순위: ``models/finbert-finetuned`` (파인튜닝 완료본) → ``FINBERT_MODEL``.
    """
    ft = finetune_model_dir()
    if (ft / "config.json").exists():
        return str(ft)
    return settings.finbert_model
