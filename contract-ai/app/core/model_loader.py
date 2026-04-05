"""
Model loader — loads the fine-tuned DistilBERT model and tokenizer once
and exposes them as module-level singletons for low-latency inference.
"""
import json
import logging
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model directory or label map is missing."""


class ModelRegistry:
    """
    Singleton that holds the tokenizer, model, and label mapping.

    Usage
    -----
    registry = ModelRegistry()
    label = registry.predict("This agreement may be terminated...")
    """

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ------------------------------------------------------------------ #
    # Public helpers                                                       #
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load tokenizer + model from disk. Call once at app startup."""
        if self._loaded:
            logger.info("ModelRegistry: already loaded, skipping.")
            return

        model_dir: Path = settings.model_dir
        label_map_path: Path = settings.label_map_path

        if not model_dir.exists():
            raise ModelNotFoundError(
                f"Model directory not found: {model_dir}. "
                "Run the training pipeline first (POST /api/v1/train)."
            )
        if not label_map_path.exists():
            raise ModelNotFoundError(
                f"Label map not found: {label_map_path}. "
                "Run the training pipeline first (POST /api/v1/train)."
            )

        logger.info("Loading tokenizer from %s …", model_dir)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(model_dir)
        )

        logger.info("Loading model from %s …", model_dir)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )

        # Choose device: CUDA → MPS (Apple Silicon) → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        # Load label mapping  { "0": "Termination", "1": "Indemnification", … }
        with open(label_map_path, "r", encoding="utf-8") as fh:
            raw: dict[str, str] = json.load(fh)
        self.id2label: dict[int, str] = {int(k): v for k, v in raw.items()}

        self._loaded = True
        logger.info(
            "Model loaded on %s — %d classes, max_length=%d",
            self.device,
            len(self.id2label),
            settings.max_length,
        )

    def unload(self) -> None:
        """Release GPU / CPU memory (called on shutdown)."""
        if not self._loaded:
            return
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("ModelRegistry: model unloaded.")


# Convenience singleton used throughout the app
model_registry = ModelRegistry()
