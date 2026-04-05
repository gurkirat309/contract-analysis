"""
Text preprocessing utilities:
  - basic text cleaning
  - HuggingFace dataset loading + train/test split
  - label encoding helpers
  - tokenization
"""
import re
import logging
from typing import Any

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Text cleaning                                                           #
# ---------------------------------------------------------------------- #

def clean_text(text: str) -> str:
    """
    Perform basic contract text cleaning:
    - Collapse excessive whitespace / newlines
    - Remove non-printable characters
    - Lowercase (DistilBERT uncased)
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[\r\n\t]+", " ", text)            # collapse line breaks
    text = re.sub(r"[^\x20-\x7E]+", " ", text)        # strip non-ASCII
    text = re.sub(r"\s{2,}", " ", text)                # collapse extra spaces
    return text.strip().lower()


# ---------------------------------------------------------------------- #
# Dataset loading                                                         #
# ---------------------------------------------------------------------- #

def load_and_split_dataset(
    dataset_name: str = settings.dataset_name,
    text_column: str = settings.text_column,
    label_column: str = settings.label_column,
    test_size: float = settings.test_size,
    seed: int = 42,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """
    Load the HuggingFace dataset, clean texts, encode labels,
    and return train/test splits plus label → id mappings.

    Returns
    -------
    splits      : DatasetDict with 'train' and 'test' keys
    label2id    : { "Termination": 0, … }
    id2label    : { 0: "Termination", … }
    """
    logger.info("Loading dataset '%s' from HuggingFace Hub …", dataset_name)
    raw: DatasetDict | Dataset = load_dataset(dataset_name)

    # Some datasets have a single 'train' split; handle both cases
    if isinstance(raw, DatasetDict) and "train" in raw:
        ds: Dataset = raw["train"]
    else:
        ds = raw  # type: ignore[assignment]

    # Validate columns
    available_cols = ds.column_names
    if text_column not in available_cols:
        raise ValueError(
            f"Text column '{text_column}' not found. Available: {available_cols}"
        )
    if label_column not in available_cols:
        raise ValueError(
            f"Label column '{label_column}' not found. Available: {available_cols}"
        )

    # --- Clean text ---
    logger.info("Cleaning text column '%s' …", text_column)
    ds = ds.map(
        lambda batch: {text_column: [clean_text(t) for t in batch[text_column]]},
        batched=True,
        desc="Cleaning text",
    )

    # --- Encode labels ---
    unique_labels: list[str] = sorted(set(ds[label_column]))
    label2id: dict[str, int] = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label: dict[int, str] = {idx: lbl for lbl, idx in label2id.items()}

    logger.info(
        "Found %d unique labels: %s …",
        len(unique_labels),
        unique_labels[:5],
    )

    ds = ds.map(
        lambda batch: {"labels": [label2id[lbl] for lbl in batch[label_column]]},
        batched=True,
        desc="Encoding labels",
    )

    # --- Train / test split ---
    splits: DatasetDict = ds.train_test_split(test_size=test_size, seed=seed)
    logger.info(
        "Split → train: %d samples, test: %d samples",
        len(splits["train"]),
        len(splits["test"]),
    )

    return splits, label2id, id2label


# ---------------------------------------------------------------------- #
# Tokenization                                                            #
# ---------------------------------------------------------------------- #

def tokenize_dataset(
    splits: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = settings.text_column,
    max_length: int = settings.max_length,
) -> DatasetDict:
    """
    Tokenize the text column in-place and remove raw text / label columns
    that the HuggingFace Trainer does not need.

    Returns a DatasetDict ready for the Trainer with columns:
    input_ids, attention_mask, labels
    """

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = splits.map(
        _tokenize,
        batched=True,
        desc="Tokenizing",
    )

    # Keep only what Trainer needs
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for split_name in tokenized:
        remove = [c for c in tokenized[split_name].column_names if c not in keep_cols]
        tokenized[split_name] = tokenized[split_name].remove_columns(remove)

    tokenized.set_format("torch")
    return tokenized
