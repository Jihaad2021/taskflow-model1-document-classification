from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings


class RuntimeConfig(BaseModel):
    """Runtime configuration (local vs kaggle, device, etc.)."""

    run_mode: Literal["local", "kaggle"] = Field(
        "local", description="Execution environment: local development or Kaggle GPU."
    )
    device: str = Field(
        "cpu",
        description="Torch device string, e.g. 'cpu' or 'cuda'. "
        "In Kaggle this will typically be 'cuda'.",
    )
    seed: int = Field(42, description="Global random seed for reproducibility.")


class DataConfig(BaseModel):
    """Configuration for dataset loading and preprocessing."""

    dataset_name: str = Field(
        "ag_news", description="HuggingFace datasets identifier for classification data."
    )
    text_column: str = Field(
        "text", description="Name of the text column in the dataset."
    )
    label_column: str = Field(
        "label", description="Name of the label column in the dataset."
    )

    max_train_samples: Optional[int] = Field(
        None,
        description="Optional cap on number of training samples "
        "(for local dev on CPU).",
    )
    max_eval_samples: Optional[int] = Field(
        None,
        description="Optional cap on number of evaluation samples "
        "(for local dev on CPU).",
    )

    def use_subset(self) -> bool:
        """Return True if any sample caps are set."""
        return self.max_train_samples is not None or self.max_eval_samples is not None


class ModelConfig(BaseModel):
    """Configuration for the sequence classification model."""

    model_name: str = Field(
        "distilbert-base-uncased",
        description="Base HuggingFace model name for classification.",
    )
    num_labels: int = Field(
        4,
        description="Number of target classes for classification "
        "(AG News has 4 labels).",
    )
    max_length: int = Field(256, description="Max tokenized sequence length.")
    dropout: float = Field(
        0.1, ge=0.0, le=1.0, description="Dropout probability for classifier head."
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters and settings."""

    epochs: int = Field(1, ge=1, description="Number of training epochs.")
    batch_size: int = Field(8, ge=1, description="Batch size for training and eval.")
    learning_rate: float = Field(
        2e-5, gt=0, description="Base learning rate for optimizer."
    )
    weight_decay: float = Field(
        0.01, ge=0.0, description="Weight decay (L2 regularization)."
    )
    warmup_steps: int = Field(
        0, ge=0, description="Number of warmup steps for learning rate scheduler."
    )
    logging_steps: int = Field(
        50, ge=1, description="Frequency (steps) for logging training metrics."
    )
    eval_steps: int = Field(
        200, ge=1, description="Frequency (steps) for evaluation during training."
    )
    save_total_limit: int = Field(
        2, ge=1, description="Maximum number of checkpoints to keep."
    )


class OptimizationConfig(BaseModel):
    """Configuration for ONNX export and optimization."""

    export_onnx: bool = Field(
        True, description="Whether to export the best model to ONNX."
    )
    onnx_opset: int = Field(17, description="ONNX opset version to use.")
    optimize_fp16: bool = Field(
        True,
        description="Whether to generate a FP16-optimized ONNX model for inference.",
    )


class APIConfig(BaseModel):
    """Configuration for the FastAPI inference service."""

    host: str = Field("0.0.0.0", description="API host address.")
    port: int = Field(8001, description="API port for Model 1 (classification).")
    reload: bool = Field(
        False, description="Enable auto-reload (dev only, not for production)."
    )


class AppConfig(BaseModel):
    """Top-level application configuration for Model 1."""

    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimization: OptimizationConfig
    api: APIConfig


class Settings(BaseSettings):
    """Environment-driven settings (secrets, paths, log level)."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = project_root / "models"
    artifacts_dir: Path = project_root / "artifacts"
    logs_dir: Path = project_root / "logs"

    # Logging
    log_level: str = "INFO"

    # Misc
    env_name: str = "dev"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def _ensure_directories(settings: Settings) -> None:
    """Ensure that important directories exist on disk.

    Args:
        settings: Settings instance containing path configuration.
    """
    for path in [settings.models_dir, settings.artifacts_dir, settings.logs_dir]:
        path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    settings = Settings()
    _ensure_directories(settings)
    return settings


def load_app_config(config_path: Path) -> AppConfig:
    """Load the application configuration from a YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        AppConfig: Parsed application configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the YAML content cannot be parsed into AppConfig.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        return AppConfig(
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            optimization=OptimizationConfig(**raw.get("optimization", {})),
            api=APIConfig(**raw.get("api", {})),
        )
    except ValidationError as exc:
        # Biarkan error ini di-handle caller dengan logging yang bagus nantinya.
        raise exc
