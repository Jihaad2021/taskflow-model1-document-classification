from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    """Minimal model-related configuration required for the data pipeline.

    At this phase, we only care about tokenizer name and max_length.
    Other training-related fields will be added in later phases.
    """

    model_name: str = Field(
        ...,
        description="HuggingFace model name used to load the tokenizer.",
    )
    max_length: int = Field(
        256,
        ge=8,
        description="Maximum sequence length for tokenization.",
    )

    # Fix warning: 'model_' protected namespace
    model_config = ConfigDict(protected_namespaces=())


class AppConfig(BaseModel):
    """Top-level application configuration for the data pipeline phase."""

    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig


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

    Only runtime, data, and model sections are expected at this phase.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    runtime_cfg = RuntimeConfig(**raw.get("runtime", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))

    return AppConfig(
        runtime=runtime_cfg,
        data=data_cfg,
        model=model_cfg,
    )
