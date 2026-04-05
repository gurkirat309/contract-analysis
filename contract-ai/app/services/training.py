"""
Training service — fine-tunes DistilBERT on the contract clause dataset
using the HuggingFace Trainer API. Results and model artifacts are saved to
the 'models/' directory so the inference service can load them.

Can be triggered programmatically via POST /api/v1/train or run standalone:
    uv run python -m app.services.training
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from app.core.config import settings
from app.services.preprocessing import load_and_split_dataset, tokenize_dataset

logger = logging.getLogger(__name__)

# Evaluation metrics loaded once at module level
_accuracy_metric = evaluate.load("accuracy")
_f1_metric = evaluate.load("f1")


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute accuracy and weighted F1 for the Trainer callback."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = _accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = _f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {
        "accuracy": accuracy["accuracy"],  # type: ignore[index]
        "f1": f1["f1"],                    # type: ignore[index]
    }


def run_training(
    num_train_epochs: int = settings.num_train_epochs,
    per_device_train_batch_size: int = settings.per_device_train_batch_size,
    per_device_eval_batch_size: int = settings.per_device_eval_batch_size,
    learning_rate: float = settings.learning_rate,
    test_size: float = settings.test_size,
) -> dict[str, Any]:
    """
    Full training pipeline:
      1. Load and preprocess the dataset
      2. Tokenize
      3. Fine-tune DistilBERT with HuggingFace Trainer
      4. Save model + label map to disk

    Parameters
    ----------
    num_train_epochs : int
    per_device_train_batch_size : int
    per_device_eval_batch_size : int
    learning_rate : float
    test_size : float
        Fraction of the dataset reserved for evaluation.

    Returns
    -------
    dict  Evaluation metrics from the best checkpoint.
    """
    # ------------------------------------------------------------------ #
    # 1. Data                                                              #
    # ------------------------------------------------------------------ #
    splits, label2id, id2label = load_and_split_dataset(test_size=test_size)

    logger.info("Initialising tokenizer: %s", settings.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)

    tokenized = tokenize_dataset(splits, tokenizer)

    # ------------------------------------------------------------------ #
    # 2. Model initialisation                                              #
    # ------------------------------------------------------------------ #
    num_labels = len(label2id)
    logger.info("Initialising model with %d labels …", num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------------------------------------------ #
    # 3. Training arguments                                               #
    # ------------------------------------------------------------------ #
    output_dir = str(settings.model_dir)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        eval_strategy=settings.eval_strategy,
        save_strategy=settings.save_strategy,
        logging_steps=settings.logging_steps,
        load_best_model_at_end=settings.load_best_model_at_end,
        metric_for_best_model=settings.metric_for_best_model,
        greater_is_better=True,
        report_to="none",            # disable W&B / HF Hub logging by default
        fp16=False,                  # set True if CUDA available for speed
        dataloader_num_workers=0,    # safe default for Windows
    )

    # ------------------------------------------------------------------ #
    # 4. Trainer                                                           #
    # ------------------------------------------------------------------ #
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------ #
    # 5. Evaluate + save                                                   #
    # ------------------------------------------------------------------ #
    logger.info("Running final evaluation …")
    eval_results: dict[str, Any] = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    logger.info("Saving model to: %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label map separately for the inference service
    label_map_path = settings.label_map_path
    os.makedirs(label_map_path.parent, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)
    logger.info("Label map saved to: %s", label_map_path)

    return eval_results


# Allow standalone execution: `uv run python -m app.services.training`
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    results = run_training()
    print("\n✅ Training complete. Metrics:")
    for k, v in results.items():
        print(f"   {k}: {v}")
