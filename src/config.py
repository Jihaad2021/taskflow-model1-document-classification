from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Dict, Optional, Any

import yaml
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================
# Runtime & Data Config (Phase: Data Pipeline)
# ============================


class RuntimeConfig(BaseModel):
    """Runtime-related settings for the current run."""

    run_mode: Literal["local", "kaggle"] = Field(
        "local",
        description="Execution mode. 'local' for dev, 'kaggle' for notebook/GPU runs.",
    )
    device: str = Field(
        "cpu",
        description="Torch device string, e.g. 'cpu', 'cuda', or 'cuda:0'.",
    )
    seed: int = Field(
        42,
        ge=0,
        description="Global random seed for reproducibility.",
    )


class DataConfig(BaseModel):
    """Dataset-related configuration."""

    dataset_name: str = Field(
        ...,
        description="HuggingFace dataset name, e.g. 'ag_news'.",
    )
    text_column: str = Field(
        ...,
        description="Name of the text column in the dataset.",
    )
    label_column: str = Field(
        ...,
        description="Name of the label column in the dataset.",
    )
    max_train_samples: int | None = Field(
        None,
        ge=1,
        description="Optional cap for number of training samples (for fast local dev).",
    )
    max_eval_samples: int | None = Field(
        None,
        ge=1,
        description="Optional cap for number of eval samples (for fast local dev).",
    )

    def use_subset(self) -> bool:
        """Whether a subset cap should be applied for local development."""
        return self.max_train_samples is not None or self.max_eval_samples is not None


class ModelConfig(BaseModel):
    """Configuration for the classification model."""

    model_name: str = Field(
        ...,
        description="HuggingFace model name, e.g. 'distilbert-base-uncased'.",
    )
    max_length: int = Field(
        256,
        description="Maximum sequence length used during tokenization.",
    )

    # New fields for Phase 2 (model-building & training)
    num_labels: Optional[int] = Field(
        None,
        description=(
            "Number of classification labels. If None, it must be provided "
            "explicitly when building the model."
        ),
    )
    id2label: Optional[Dict[int, str]] = Field(
        default=None,
        description="Optional mapping from class index to label name.",
    )
    label2id: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional mapping from label name to class index.",
    )

class TrainingConfig(BaseModel):
    """Configuration for model training (Phase 2)."""

    output_dir_name: str = Field(
        "model_1_classifier",
        description="Subdirectory name under models_dir where checkpoints will be saved.",
    )
    num_train_epochs: int = Field(
        3,
        ge=1,
        description="Number of training epochs.",
    )
    per_device_train_batch_size: int = Field(
        16,
        ge=1,
        description="Training batch size per device.",
    )
    per_device_eval_batch_size: int = Field(
        32,
        ge=1,
        description="Evaluation batch size per device.",
    )
    learning_rate: float = Field(
        5e-5,
        gt=0.0,
        description="Learning rate for the optimizer.",
    )
    weight_decay: float = Field(
        0.0,
        ge=0.0,
        description="Weight decay for optimizer.",
    )
    warmup_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Warmup ratio for the learning rate scheduler.",
    )
    logging_steps: int = Field(
        50,
        ge=1,
        description="Interval (in steps) for logging training metrics.",
    )
    evaluation_strategy: str = Field(
        "epoch",
        description="Evaluation strategy used by HF Trainer (e.g. 'no', 'steps', 'epoch').",
    )
    save_strategy: str = Field(
        "epoch",
        description="Checkpoint save strategy used by HF Trainer.",
    )
    metric_for_best_model: str = Field(
        "f1",
        description="Primary metric used to select the best model.",
    )
    greater_is_better: bool = Field(
        True,
        description="Whether a higher value of metric_for_best_model is better.",
    )

class AppConfig(BaseModel):
    """Top-level application configuration for the data pipeline phase."""

    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

# ============================
# Settings (paths, env) – infra-level, not phase-specific
# ============================


class Settings(BaseSettings):
    """Application settings not tied to a specific phase config.

    These are mostly paths, logging level, and other environment-level settings.
    """

    log_level: str = "INFO"

    # Project root: <repo_root> (assumes src/ is under this)
    project_root: Path = Path(__file__).resolve().parents[1]

    # Common directories
    configs_dir: Path = project_root / "configs"
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = project_root / "artifacts"
    models_dir: Path = project_root / "models"

    # Fix Pydantic v2 config-style deprecation
    model_config = SettingsConfigDict(
        env_prefix="TASKFLOW_M1_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


# ============================
# Loader
# ============================


def load_app_config(config_path: str | Path) -> AppConfig:
    """Load application configuration from a YAML file.

    The following sections are expected:
        - runtime
        - data
        - model
        - training (optional; defaults will be used if missing)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    runtime_cfg = RuntimeConfig(**raw.get("runtime", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))

    return AppConfig(
        runtime=runtime_cfg,
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
    )
