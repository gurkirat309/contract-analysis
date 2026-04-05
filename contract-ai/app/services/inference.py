"""
Inference service — wraps the loaded model to run fast clause classification.
"""
import logging
import time
from typing import NamedTuple

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.core.model_loader import model_registry
from app.services.preprocessing import clean_text

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    label: str
    confidence: float
    latency_ms: float


def predict_clause(text: str) -> PredictionResult:
    """
    Classify a single contract clause and return its label + confidence.

    Parameters
    ----------
    text : str
        Raw contract clause string (up to 10 000 chars — will be truncated
        to ``settings.max_length`` tokens internally).

    Returns
    -------
    PredictionResult
        Named tuple with ``label`` (str), ``confidence`` (float 0–1),
        and ``latency_ms`` (float, wall-clock inference time in ms).

    Raises
    ------
    RuntimeError
        If the model registry has not been loaded yet.
    """
    if not model_registry.is_loaded:
        raise RuntimeError(
            "Model is not loaded. The server may still be starting up."
        )

    registry = model_registry
    t0 = time.perf_counter()

    # 1. Clean input
    clean = clean_text(text)

    # 2. Tokenize
    inputs = registry.tokenizer(
        clean,
        truncation=True,
        padding="max_length",
        max_length=settings.max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(registry.device) for k, v in inputs.items()}

    # 3. Forward pass (no gradient tracking needed)
    with torch.no_grad():
        outputs = registry.model(**inputs)
        logits: torch.Tensor = outputs.logits  # shape: (1, num_classes)

    # 4. Convert to probabilities
    probs: torch.Tensor = F.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)
    pred_id: int = int(probs.argmax().item())
    confidence: float = float(probs[pred_id].item())

    label: str = registry.id2label.get(pred_id, f"UNKNOWN_{pred_id}")

    latency_ms = (time.perf_counter() - t0) * 1_000
    logger.debug(
        "predict_clause → label='%s', conf=%.4f, latency=%.1f ms",
        label, confidence, latency_ms,
    )

    return PredictionResult(label=label, confidence=confidence, latency_ms=latency_ms)
