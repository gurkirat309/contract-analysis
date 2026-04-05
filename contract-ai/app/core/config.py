"""
Application configuration loaded from environment variables or .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Resolve the project root (two levels up from this file: core/ -> app/ -> contract-ai/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central application settings."""

    # --- Application ---
    app_name: str = "Contract Analysis AI"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")

    # --- Model --
    base_model_name: str = Field(
        default="distilbert-base-uncased",
        description="HuggingFace model ID used for fine-tuning",
    )
    max_length: int = Field(default=512, description="Max tokenization length")
    model_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "contract_classifier",
        description="Path to save / load the fine-tuned model",
    )
    label_map_path: Path = Field(
        default=PROJECT_ROOT / "models" / "label_map.json",
        description="Path to the label-id JSON mapping file",
    )

    # --- Training ---
    dataset_name: str = "alisha4walunj/quad_ledgar_merged_dataset"
    text_column: str = "text"  # column containing clause text
    label_column: str = "label"     # column containing the class label
    test_size: float = 0.15
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"

    # --- Data ---
    data_dir: Path = PROJECT_ROOT / "data"

    # --- API ---
    api_prefix: str = "/api/v1"

    class Config:
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance (imported everywhere)
settings = Settings()
